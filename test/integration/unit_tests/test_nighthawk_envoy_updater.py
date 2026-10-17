"""Unit tests for nighthawk_envoy_updater checkpointing and step decorators."""

import enum
import pathlib
import subprocess
from unittest import mock
import pytest

from tools import nighthawk_envoy_updater as updater


class _SampleStep(enum.Enum):
  """Sample steps for testing StepHandler registration and checkpointing."""

  STEP_ONE = enum.auto()
  STEP_TWO = enum.auto()
  STEP_THREE = enum.auto()


class _SampleWorkflow(updater.StepHandler[_SampleStep]):
  """Sample workflow using @register_step and @checkpointed decorators."""

  def __init__(
      self,
      checkpoint_store: updater.CheckpointStore,
      fail_step_two: bool = False,
  ) -> None:
    """Initialize the sample workflow.

    Args:
      checkpoint_store: CheckpointStore used for step persistence.
      fail_step_two: Whether STEP_TWO should raise a RuntimeError.
    """
    super().__init__(
        _SampleStep,
        interactive=False,
        checkpoint_store=checkpoint_store,
    )
    self.fail_step_two = fail_step_two
    self.executed_steps: list[_SampleStep] = []
    self.expensive_calls: list[str] = []
    self._restore_checkpointed_steps()

  @updater.checkpointed(lambda self, label: f"op:{label}")
  def compute_expensive_value(self, label: str) -> str:
    """Run an operation whose result is persisted in checkpoint_store.

    Args:
      label: Identifier for the operation.

    Returns:
      Computed string result.
    """
    self.expensive_calls.append(label)
    return f"result:{label}"

  @updater.register_step(_SampleStep.STEP_ONE, checkpoint=True)
  def _run_step_one(self) -> None:
    """Execute STEP_ONE."""
    self.executed_steps.append(_SampleStep.STEP_ONE)

  @updater.register_step(_SampleStep.STEP_TWO, checkpoint=True)
  def _run_step_two(self) -> None:
    """Execute STEP_TWO, optionally failing to simulate mid-run interruption."""
    self.executed_steps.append(_SampleStep.STEP_TWO)
    if self.fail_step_two:
      raise RuntimeError("Simulated failure in STEP_TWO")

  @updater.register_step(_SampleStep.STEP_THREE, checkpoint=True)
  def _run_step_three(self) -> None:
    """Execute STEP_THREE."""
    self.executed_steps.append(_SampleStep.STEP_THREE)

  def run(self) -> dict[_SampleStep, updater.StepResult]:
    """Run all steps in the sample workflow.

    Returns:
      Mapping of _SampleStep to StepResult.
    """
    return self._run_steps()


def test_checkpoint_store_roundtrip_and_corrupt_file_recovery(tmp_path: pathlib.Path,) -> None:
  """Verify CheckpointStore saves atomically, reloads, and handles bad JSON."""
  state_path = tmp_path / "state.json"
  store = updater.CheckpointStore.load(state_path)
  store.base_envoy_commit = "a" * 40
  store.target_envoy_commit = "b" * 40
  store.completed["step:STEP_ONE"] = True
  store.save()

  reloaded = updater.CheckpointStore.load(state_path)
  assert reloaded.base_envoy_commit == "a" * 40
  assert reloaded.target_envoy_commit == "b" * 40
  assert reloaded.completed == {"step:STEP_ONE": True}

  # Corrupt the state file and verify load falls back to an empty store.
  state_path.write_text("{corrupt json", encoding="utf-8")
  recovered = updater.CheckpointStore.load(state_path)
  assert recovered.base_envoy_commit is None
  assert recovered.target_envoy_commit is None
  assert not recovered.completed


def test_checkpoint_store_bind_target_invalidates_stale_target_preserving_bisect(
    tmp_path: pathlib.Path,) -> None:
  """Verify bind_target clears target steps on mismatch while keeping bisect keys."""
  base_sha = "1" * 40
  target_a = "a" * 40
  target_b = "b" * 40
  store = updater.CheckpointStore(
      path=tmp_path / "state.json",
      base_envoy_commit=base_sha,
      target_envoy_commit=target_a,
      completed={
          f"step:{target_a}:COPY_EXACT_FILES": True,
          f"merge:{target_a}:.bazelrc": None,
          f"bisect:{base_sha}..{target_a}": True,
      },
  )

  # Matching (base, target, working_tree) retains all target checkpoints.
  store.bind_target(base_sha, target_a, working_tree_commit=target_a)
  assert f"step:{target_a}:COPY_EXACT_FILES" in store.completed

  # Switching to target_b clears target_a keys while keeping bisect verdicts.
  store.bind_target(base_sha, target_b, working_tree_commit=target_b)
  assert store.target_envoy_commit == target_b
  assert f"step:{target_a}:COPY_EXACT_FILES" not in store.completed
  assert f"merge:{target_a}:.bazelrc" not in store.completed
  assert bool(store.completed.get(f"bisect:{base_sha}..{target_a}"))


def test_checkpointed_decorator_memoizes_across_reloads(tmp_path: pathlib.Path,) -> None:
  """Verify @checkpointed persists return values and skips duplicate calls."""
  state_path = tmp_path / "state.json"
  store1 = updater.CheckpointStore.load(state_path)
  workflow1 = _SampleWorkflow(checkpoint_store=store1)

  assert workflow1.compute_expensive_value("alpha") == "result:alpha"
  assert workflow1.compute_expensive_value("alpha") == "result:alpha"
  assert workflow1.expensive_calls == ["alpha"]

  # Reload from disk in a new workflow instance and verify no re-execution.
  store2 = updater.CheckpointStore.load(state_path)
  workflow2 = _SampleWorkflow(checkpoint_store=store2)
  assert workflow2.compute_expensive_value("alpha") == "result:alpha"
  assert not workflow2.expensive_calls


def test_register_step_decorator_resumes_from_failed_step(tmp_path: pathlib.Path,) -> None:
  """Verify @register_step skips completed steps when resuming after failure."""
  state_path = tmp_path / "state.json"
  store1 = updater.CheckpointStore.load(state_path)
  run1 = _SampleWorkflow(checkpoint_store=store1, fail_step_two=True)
  results1 = run1.run()

  assert results1[_SampleStep.STEP_ONE].step_status == updater.StepStatus.SUCCESS
  assert results1[_SampleStep.STEP_TWO].step_status == updater.StepStatus.FAILED
  assert (results1[_SampleStep.STEP_THREE].step_status == updater.StepStatus.CANCELLED)
  assert run1.executed_steps == [_SampleStep.STEP_ONE, _SampleStep.STEP_TWO]

  # Resume with a new instance after fixing the failure condition.
  store2 = updater.CheckpointStore.load(state_path)
  run2 = _SampleWorkflow(checkpoint_store=store2, fail_step_two=False)
  results2 = run2.run()

  assert updater.is_clean_results(results2)
  assert run2.executed_steps == [_SampleStep.STEP_TWO, _SampleStep.STEP_THREE]


def test_patch_shared_files_conflict_resolution_is_idempotent_across_reruns(
    tmp_path: pathlib.Path,) -> None:
  """Verify resolving a shared-file conflict and re-running does not re-merge."""
  envoy_dir = tmp_path / "envoy"
  nighthawk_dir = tmp_path / "nighthawk"
  envoy_dir.mkdir(parents=True)
  nighthawk_dir.mkdir(parents=True)

  base_envoy_commit = "a" * 40
  target_envoy_commit = "b" * 40
  old_reg_sha = "1" * 40
  new_reg_sha = "2" * 40

  for rel_path in updater.COPIED_FILES + updater.SHARED_FILES:
    for root in (envoy_dir, nighthawk_dir):
      full_path = root / rel_path
      full_path.parent.mkdir(parents=True, exist_ok=True)
      full_path.write_text(f"# content {rel_path}\n", encoding="utf-8")

  (nighthawk_dir / "MODULE.bazel").write_text(
      f'ENVOY_COMMIT = "{base_envoy_commit}"\n',
      encoding="utf-8",
  )
  base_bazelrc = (
      "common"
      f" --registry=https://raw.githubusercontent.com/envoyproxy/bazel-registry/{old_reg_sha}/\n"
      "build --copt=-O2\n")
  target_bazelrc = (
      "common"
      f" --registry=https://raw.githubusercontent.com/envoyproxy/bazel-registry/{new_reg_sha}/\n"
      "build --copt=-O3\n")
  (nighthawk_dir / ".bazelrc").write_text(
      "common"
      f" --registry=https://raw.githubusercontent.com/envoyproxy/bazel-registry/{old_reg_sha}/\n"
      "build --copt=-O2 # unique\n",
      encoding="utf-8",
  )

  merge_file_invocations: list[str] = []

  def fake_run_command(
      command: list[str],
      cwd: pathlib.Path | None = None,
      input_str: str | None = None,
      shell: bool = False,
      interactive: bool = False,
  ) -> str:
    """Simulate git, cp, and sed commands hermetically without external git."""
    del cwd, input_str, shell, interactive
    if command[:2] == ["git", "show"]:
      spec = command[2]
      if spec == f"{base_envoy_commit}:.bazelrc":
        return base_bazelrc.strip()
      if spec == f"{target_envoy_commit}:.bazelrc":
        return target_bazelrc.strip()
      return "# unchanged"
    if command[:2] == ["git", "diff"]:
      shared_file = command[-1]
      return "diff" if shared_file == ".bazelrc" else ""
    if command[0] == "sed":
      module_path = pathlib.Path(command[-1])
      module_path.write_text(
          f'ENVOY_COMMIT = "{target_envoy_commit}"\n',
          encoding="utf-8",
      )
      return ""
    return ""

  def fake_subprocess_run(command: list[str], *args: object,
                          **kwargs: object) -> subprocess.CompletedProcess[str]:
    """Simulate git merge-file writing conflict markers into .bazelrc."""
    del args, kwargs
    if command[:2] == ["git", "merge-file"]:
      target_path = pathlib.Path(command[2])
      merge_file_invocations.append(target_path.name)
      target_path.write_text(
          "common"
          f" --registry=https://raw.githubusercontent.com/envoyproxy/bazel-registry/{old_reg_sha}/\n"
          "<<<<<<< .bazelrc\n"
          "build --copt=-O2 # unique\n"
          "=======\n"
          "build --copt=-O3\n"
          ">>>>>>> target\n",
          encoding="utf-8",
      )
      return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

  state_path = tmp_path / "nighthawk_envoy_updater_state.json"
  with (
      mock.patch.object(updater, "_run_command", side_effect=fake_run_command, autospec=True),
      mock.patch.object(
          updater.subprocess,
          "run",
          side_effect=fake_subprocess_run,
          autospec=True,
      ),
  ):
    store1 = updater.CheckpointStore.load(state_path)
    integration1 = updater.EnvoyCommitIntegration(
        nighthawk_git_repo_dir=nighthawk_dir,
        envoy_git_repo_dir=envoy_dir,
        current_envoy_commit=base_envoy_commit,
        target_envoy_commit=target_envoy_commit,
        interactive=False,
        checkpoint_store=store1,
    )
    for step in list(updater.EnvoyCommitIntegrationStep)[4:]:
      integration1.set_step_not_planned(step)

    results1 = integration1.run_envoy_commit_integration_steps()
    assert (results1[updater.EnvoyCommitIntegrationStep.PATCH_SHARED_FILES].step_status ==
            updater.StepStatus.FAILED)
    assert merge_file_invocations == [".bazelrc"]
    assert updater.has_conflict_markers(nighthawk_dir / ".bazelrc")

    # Simulate user resolving the conflict in .bazelrc.
    (nighthawk_dir / ".bazelrc").write_text(
        "common"
        f" --registry=https://raw.githubusercontent.com/envoyproxy/bazel-registry/{old_reg_sha}/\n"
        "build --copt=-O3 # unique\n",
        encoding="utf-8",
    )

    # Re-run a fresh EnvoyCommitIntegration loaded from the checkpoint file.
    store2 = updater.CheckpointStore.load(state_path)
    integration2 = updater.EnvoyCommitIntegration(
        nighthawk_git_repo_dir=nighthawk_dir,
        envoy_git_repo_dir=envoy_dir,
        current_envoy_commit=base_envoy_commit,
        target_envoy_commit=target_envoy_commit,
        interactive=False,
        checkpoint_store=store2,
    )
    for step in list(updater.EnvoyCommitIntegrationStep)[4:]:
      integration2.set_step_not_planned(step)

    results2 = integration2.run_envoy_commit_integration_steps()
    assert updater.is_clean_results(results2)
    # git merge-file must NOT have been called a second time!
    assert merge_file_invocations == [".bazelrc"]
    final_bazelrc = (nighthawk_dir / ".bazelrc").read_text(encoding="utf-8")
    assert new_reg_sha in final_bazelrc
    assert "build --copt=-O3 # unique" in final_bazelrc


if __name__ == "__main__":
  raise SystemExit(pytest.main([__file__]))
