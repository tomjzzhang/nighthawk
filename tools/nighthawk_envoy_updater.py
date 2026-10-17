#!/usr/bin/env python3
"""Updates the Envoy dependency in the Nighthawk repository.

This script automates the process of updating the Envoy commit hash in the
Nighthawk project. It finds the latest Envoy commit that passes Nighthawk's
tests, and if successful, creates a pull request with the update. If there is
a non-trivial Envoy merge, it leaves the local Nighthawk git repo in the
simplest compromised state and prompts the user to resolve merge conflicts,
rebase, and re-run the utility to proceed.
"""

import argparse
from collections.abc import Callable
import dataclasses
import datetime
import enum
import functools
import json
import logging
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback
from typing import Any, Generic, TypeVar

MAX_AGENT_ATTEMPTS = 3

# Files Nighthawk re-uses verbatim from its Envoy dependency.
COPIED_FILES: list[str] = [
    ".bazelversion",
    ".github/config.yml",
    "ci/envoy_build_sha.sh",
    "ci/run_envoy_docker.sh",
]

# Files Nighthawk re-uses (almost) verbatim from its Envoy dependency.
SHARED_FILES: list[str] = [
    ".bazelrc",
    "tools/gen_compilation_database.py",
    "tools/code_format/config.yaml",
    "ci/docker-compose.yml",
]

_BAZEL_REGISTRY_URL_PATTERN = re.compile(
    r"(raw\.githubusercontent\.com/envoyproxy/bazel-registry/)([0-9a-fA-F]{40})")
_BAZEL_DEP_PATTERN = re.compile(
    r'^bazel_dep\(\s*name\s*=\s*"([^"]+)",\s*version\s*=\s*"([^"]+)"',
    re.MULTILINE,
)
_ENVOY_MODULE_VERSION_PATTERN = re.compile(
    r'module\(\s*name\s*=\s*"envoy",\s*version\s*=\s*"([^"]+)"',
    re.DOTALL,
)
_ENVOY_COMMIT_PATTERN = re.compile(
    r'^ENVOY_COMMIT = "([0-9a-fA-F]{40})"$',
    re.MULTILINE,
)
_LAST_UPDATED_PATTERN = re.compile(
    r"^# Last updated .*$",
    re.MULTILINE,
)
_CONFLICT_MARKER_PATTERN = re.compile(
    r"^(<<<<<<<|=======|>>>>>>>)( .*)?$",
    re.MULTILINE,
)

TStep = TypeVar("TStep", bound=enum.Enum)
TResult = TypeVar("TResult")


def _run_command(
    command: list[str],
    cwd: pathlib.Path | None = None,
    input_str: str | None = None,
    shell: bool = False,
    interactive: bool = False,
) -> str:
  """Run a shell command and return the standard output.

  Args:
    command: The command to run as a list of strings.
    cwd: The working directory for the command.
    input_str: Input to pass to stdin.
    shell: Whether to run the command in a shell.
    interactive: Whether to prompt the user before running the command.

  Returns:
    The standard output of the command.
  """
  if interactive:
    input("Press enter and monitor for ~10 seconds for more prompts...")
  cmd = f"{cwd if cwd else pathlib.Path.cwd()} $ {shlex.join(command)}"
  logging.info("%s", cmd)
  try:
    result = subprocess.run(
        command,
        cwd=cwd,
        input=input_str,
        shell=shell,
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout:
      logging.debug("Stdout:\n%s", result.stdout.strip())
    if result.stderr:
      logging.debug("Stderr:\n%s", result.stderr.strip())
    return result.stdout.strip()
  except subprocess.CalledProcessError as e:
    logging.error(
        "Command '%s' failed with exit code %s",
        shlex.join(command),
        e.returncode,
    )
    e.cmd = cmd
    raise e


def _run_sed_replace(
    old_pattern: str,
    new_pattern: str,
    filename: str,
    cwd: pathlib.Path | None = None,
) -> None:
  """Run a sed command to replace a pattern in a file in-place.

  Args:
    old_pattern: The regex pattern to search for.
    new_pattern: The replacement string.
    filename: The file to modify.
    cwd: The working directory for the command.
  """
  _run_command(
      ["sed", "-i", f"s/{old_pattern}/{new_pattern}/g", filename],
      cwd=cwd,
  )


def has_conflict_markers(file_path: pathlib.Path) -> bool:
  """Return whether a file contains unresolved git conflict markers.

  Args:
    file_path: Path to the file to inspect.

  Returns:
    True if standard conflict markers are present in the file.
  """
  if not file_path.exists():
    return False
  return bool(_CONFLICT_MARKER_PATTERN.search(file_path.read_text(encoding="utf-8")))


@dataclasses.dataclass
class CheckpointStore:
  """Persists completed checkpoint keys and results to a JSON file.

  Attributes:
    path: Path to the JSON state file on disk.
    base_envoy_commit: The starting Envoy commit in Nighthawk's HEAD.
    target_envoy_commit: The target Envoy commit currently being integrated.
    completed: Mapping of checkpoint keys to their stored results.
  """

  path: pathlib.Path
  base_envoy_commit: str | None = None
  target_envoy_commit: str | None = None
  completed: dict[str, Any] = dataclasses.field(default_factory=dict)

  @classmethod
  def load(cls, path: pathlib.Path) -> "CheckpointStore":
    """Load checkpoint state from disk if present and valid.

    Args:
      path: Path to the JSON checkpoint file.

    Returns:
      A loaded or freshly initialized CheckpointStore instance.
    """
    if not path.exists():
      return cls(path=path)
    try:
      data = json.loads(path.read_text(encoding="utf-8"))
      if not isinstance(data, dict):
        return cls(path=path)
      completed = data.get("completed", {})
      if not isinstance(completed, dict):
        completed = {}
      return cls(
          path=path,
          base_envoy_commit=data.get("base_envoy_commit"),
          target_envoy_commit=data.get("target_envoy_commit"),
          completed=completed,
      )
    except (OSError, json.JSONDecodeError, TypeError) as e:
      logging.warning("Ignoring unreadable checkpoint file %s: %s", path, e)
      return cls(path=path)

  def save(self) -> None:
    """Atomically write the checkpoint state to disk."""
    self.path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_envoy_commit": self.base_envoy_commit,
        "target_envoy_commit": self.target_envoy_commit,
        "completed": self.completed,
    }
    tmp_path = self.path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(self.path)

  def reset(self) -> None:
    """Clear all in-memory state and remove the checkpoint file on disk."""
    self.base_envoy_commit = None
    self.target_envoy_commit = None
    self.completed.clear()
    if self.path.exists():
      self.path.unlink()

  def bind_target(
      self,
      base_commit: str,
      target_commit: str,
      working_tree_commit: str | None,
  ) -> None:
    """Bind the checkpoint store to a specific (base, target) integration.

    If the stored base/target commits do not match or the working tree's
    MODULE.bazel is not currently prepared for target_commit, any target-scoped
    checkpoints are invalidated while preserving bisect verdicts for the same
    base_commit.

    Args:
      base_commit: The starting Envoy commit hash in Nighthawk.
      target_commit: The target Envoy commit hash being integrated.
      working_tree_commit: The ENVOY_COMMIT currently in working-tree
        MODULE.bazel.
    """
    is_same_target = (self.base_envoy_commit == base_commit and
                      self.target_envoy_commit == target_commit and
                      working_tree_commit == target_commit)
    if is_same_target:
      return

    preserved: dict[str, Any] = {}
    if self.base_envoy_commit == base_commit:
      preserved = {
          k: v for k, v in self.completed.items() if k.startswith(f"bisect:{base_commit}..")
      }
    self.base_envoy_commit = base_commit
    self.target_envoy_commit = target_commit
    self.completed = preserved
    self.save()

  def clear_target(self, target_commit: str) -> None:
    """Remove all step and merge checkpoints for a target commit.

    Args:
      target_commit: The target Envoy commit hash whose state should be cleared.
    """
    prefixes = (f"step:{target_commit}:", f"merge:{target_commit}:")
    self.completed = {k: v for k, v in self.completed.items() if not k.startswith(prefixes)}
    self.save()


def checkpointed(
    key_fn: Callable[..., str],) -> Callable[[Callable[..., TResult]], Callable[..., TResult]]:
  """Memoize a method's execution in self.checkpoint_store.

  Args:
    key_fn: Callable taking (self, *args, **kwargs) and returning a unique
      checkpoint key string.

  Returns:
    A decorator wrapping the target method with CheckpointStore lookup/save.
  """

  def decorator(fn: Callable[..., TResult]) -> Callable[..., TResult]:
    """Wrap the method with checkpoint lookup and persistence."""

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> TResult:
      """Return cached checkpoint value if present, else execute and save."""
      store: CheckpointStore | None = getattr(self, "checkpoint_store", None)
      if store is None:
        return fn(self, *args, **kwargs)
      key = key_fn(self, *args, **kwargs)
      if key in store.completed:
        logging.info("Skipping checkpointed operation: %s", key)
        return store.completed[key]
      result = fn(self, *args, **kwargs)
      store.completed[key] = result
      store.save()
      return result

    return wrapper

  return decorator


def register_step(
    step_enum_member: enum.Enum,
    *,
    checkpoint: bool = False,
) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
  """Bind a method to a StepHandler enum step.

  Args:
    step_enum_member: The enum member corresponding to the decorated method.
    checkpoint: Whether completion of this step should be persisted in
      self.checkpoint_store.

  Returns:
    A decorator attaching step registration metadata to the method.
  """

  def decorator(fn: Callable[[Any], None]) -> Callable[[Any], None]:
    """Attach step enum and checkpoint metadata to the method."""
    setattr(fn, "_step_enum", step_enum_member)
    setattr(fn, "_checkpoint_step", checkpoint)
    return fn

  return decorator


class StepStatus(enum.Enum):
  """Status of an individual step in the update process."""

  PENDING = enum.auto()
  SUCCESS = enum.auto()
  FAILED = enum.auto()
  CANCELLED = enum.auto()
  NOT_PLANNED = enum.auto()


@dataclasses.dataclass
class StepResult:
  """Result of a step execution, including status and error details.

  Attributes:
    step_status: The current status of the step.
    cmd: The command that was executed, if applicable.
    stdout: The standard output from the command.
    stderr: The standard error from the command.
    traceback: The exception traceback if the step failed.
    step_handler_state: A snapshot of the StepHandler's state variables.
  """

  step_status: StepStatus
  cmd: str | None
  stdout: str | None
  stderr: str | None
  traceback: str | None
  step_handler_state: dict[str, Any] | None


def is_clean_results(results: dict[Any, StepResult]) -> bool:
  """Check if all steps in the results dictionary succeeded or were not planned.

  Args:
    results: A dictionary mapping steps to their StepResult.

  Returns:
    True if all steps are either SUCCESS or NOT_PLANNED, False otherwise.
  """
  return all(result.step_status in [StepStatus.SUCCESS, StepStatus.NOT_PLANNED]
             for result in results.values())


def format_step_result(result: StepResult, print_internal_state: bool = False) -> str:
  """Format a single StepResult into a human-readable string.

  Args:
    result: The StepResult to format.
    print_internal_state: Whether to include the internal state of the
      StepHandler.

  Returns:
    A formatted string representing the StepResult.
  """
  output = []
  output.append(f"  Status: {result.step_status.name}")
  if print_internal_state and result.step_handler_state:
    output.append("  Internal State:")
    for k, v in result.step_handler_state.items():
      output.append(f"    {k}: {v}")
  if result.cmd:
    output.append(f"  Command: {result.cmd}")
  if result.stdout:
    output.append("  Stdout:")
    for line in result.stdout.strip().splitlines():
      output.append(f"    {line}")
  if result.stderr:
    output.append("  Stderr:")
    for line in result.stderr.strip().splitlines():
      output.append(f"    {line}")
  if result.traceback:
    output.append("  Traceback:")
    for line in result.traceback.strip().splitlines():
      output.append(f"    {line}")
  return "\n".join(output)


def format_step_results(results: dict[TStep, StepResult]) -> str:
  """Format step results into a human-readable string.

  Args:
    results: Mapping of step enum values to their StepResult.

  Returns:
    A formatted multi-line summary string.
  """
  output = []
  for step, result in results.items():
    output.append(f"Step: {step.name}")
    output.append(format_step_result(result))
    output.append("-" * 20)
  return "\n".join(output)


class StepHandler(Generic[TStep]):
  """Base class for handling a sequence of steps registered via @register_step.

  Attributes:
    steps: A list of all possible steps in execution order.
    step_tracker: A dictionary to track the status and errors of each step.
    agent_invocation: Command template to invoke an LLM agent to fix step
      failures.
    interactive: Whether to pause and prompt the user when a step fails.
    checkpoint_store: Optional CheckpointStore for persisting checkpointed
      steps.
  """

  def __init__(
      self,
      step_enum: type[TStep],
      agent_invocation: str | None = None,
      interactive: bool = True,
      checkpoint_store: CheckpointStore | None = None,
  ) -> None:
    """Initialize the StepHandler and discover @register_step methods.

    Args:
      step_enum: The enum type defining the sequence of steps.
      agent_invocation: Command template to invoke the agent.
      interactive: Whether to pause and prompt the user when a step fails.
      checkpoint_store: Optional CheckpointStore used by @register_step and
        @checkpointed.
    """
    self.steps = list(step_enum)
    self.step_tracker = {
        step:
            StepResult(
                step_status=StepStatus.PENDING,
                cmd=None,
                stdout=None,
                stderr=None,
                traceback=None,
                step_handler_state=None,
            ) for step in self.steps
    }
    self.agent_invocation = agent_invocation
    self.interactive = interactive
    self.checkpoint_store = checkpoint_store

    self._step_handlers: dict[TStep, Callable[[], None]] = {}
    self._checkpointed_steps: set[TStep] = set()
    for attr_name in dir(self):
      attr = getattr(self, attr_name, None)
      step_member = getattr(attr, "_step_enum", None)
      if isinstance(step_member, step_enum):
        self._step_handlers[step_member] = attr
        if getattr(attr, "_checkpoint_step", False):
          self._checkpointed_steps.add(step_member)

  def _step_checkpoint_key(self, step: TStep) -> str:
    """Return the CheckpointStore key for a given step.

    Args:
      step: The step enum member.

    Returns:
      A string key uniquely identifying the step within the current scope.
    """
    return f"step:{step.name}"

  def _restore_checkpointed_steps(self) -> None:
    """Mark any already-checkpointed steps as SUCCESS in step_tracker."""
    if self.checkpoint_store is None:
      return
    for step in self.steps:
      if step in self._checkpointed_steps:
        key = self._step_checkpoint_key(step)
        if bool(self.checkpoint_store.completed.get(key)):
          self.step_tracker[step].step_status = StepStatus.SUCCESS

  def _set_step_success(self, step: TStep) -> None:
    """Set the status of a step to SUCCESS and persist if checkpointed.

    Args:
      step: The step that succeeded.
    """
    self.step_tracker[step] = StepResult(
        step_status=StepStatus.SUCCESS,
        cmd=None,
        stdout=None,
        stderr=None,
        traceback=None,
        step_handler_state=None,
    )
    for member in self.steps:
      if self.step_tracker[member].step_status == StepStatus.CANCELLED:
        self.step_tracker[member].step_status = StepStatus.PENDING
    if self.checkpoint_store is not None and step in self._checkpointed_steps:
      self.checkpoint_store.completed[self._step_checkpoint_key(step)] = True
      self.checkpoint_store.save()

  def _set_step_pending(self, step: TStep) -> None:
    """Reset a step and any cancelled steps to PENDING.

    Args:
      step: The step to reset to pending.
    """
    self.step_tracker[step] = StepResult(
        step_status=StepStatus.PENDING,
        cmd=None,
        stdout=None,
        stderr=None,
        traceback=None,
        step_handler_state=None,
    )
    for member in self.steps:
      if self.step_tracker[member].step_status == StepStatus.CANCELLED:
        self.step_tracker[member].step_status = StepStatus.PENDING

  def _set_step_failure(self, step: TStep, error: Exception) -> None:
    """Set the status of a step to FAILED and record the error.

    Args:
      step: The step that failed.
      error: The exception that caused the failure.
    """
    step_result = self.step_tracker[step]
    step_result.step_status = StepStatus.FAILED
    step_result.cmd = getattr(error, "cmd", None)
    step_result.stdout = getattr(error, "stdout", None)
    step_result.stderr = getattr(error, "stderr", None)
    step_result.traceback = "\n".join(traceback.format_exception(error))
    step_result.step_handler_state = self._get_print_vars()

    logging.error("%s", format_step_result(step_result, print_internal_state=True))

    for member in self.steps:
      if self.step_tracker[member].step_status == StepStatus.PENDING:
        self.step_tracker[member].step_status = StepStatus.CANCELLED

  def set_step_not_planned(self, step: TStep) -> None:
    """Set the status of a step to NOT_PLANNED.

    Args:
      step: The step to mark as not planned.
    """
    self.step_tracker[step].step_status = StepStatus.NOT_PLANNED

  def _set_step_not_planned(self, step: TStep) -> None:
    """Set the status of a step to NOT_PLANNED.

    Args:
      step: The step to mark as not planned.
    """
    self.set_step_not_planned(step)

  def _get_print_vars(self) -> dict[str, Any]:
    """Build a dictionary of instance variables to print for debugging.

    Returns:
      A dictionary of instance attributes excluding internal handler tables.
    """
    print_vars = vars(self).copy()
    for excluded in (
        "steps",
        "step_tracker",
        "_step_handlers",
        "_checkpointed_steps",
        "checkpoint_store",
    ):
      print_vars.pop(excluded, None)
    return print_vars

  def _create_agent_prompt(self, step: TStep) -> str:
    """Create a prompt string for the agent when a step fails.

    Args:
      step: The step that failed.

    Returns:
      A markdown-formatted prompt describing the failure and context.
    """
    return "\n".join([
        "## Step handler state and tracker",
        "",
        f"{format_step_result(self.step_tracker[step], True)}",
        "",
        "## Agent Instructions",
        "",
        "Based on the context and instance state above, fix the files in the ",
        "nighthawk_git_repo_dir to resolve the error. The primary goal ",
        "is to make the failing step pass.\n",
        "Avoid changing the behavior of Nighthawk's tests, if possible. ",
        "Avoid changing compilation flags, if possible. ",
        "The Envoy source code for the target commit is available in a ",
        "directory under the Nighthawk project root parent, ",
        "envoy_git_repo_dir.\n\n",
        "Do **not** modify the Nighthawk git repo state (add, commit, etc.)",
    ])

  def _prompt_interactive_failure(self, step: TStep, error: Exception) -> str:
    """Prompt the user to resolve a step failure interactively.

    Args:
      step: The step that failed.
      error: The exception that caused the failure.

    Returns:
      One of 'retry', 'proceed', or 'exit'.
    """
    print("\n" + "=" * 78)
    print(f"STEP FAILED: {step.name}")
    print("=" * 78)
    cmd = getattr(error, "cmd", None)
    if cmd:
      print(f"Command: {cmd}")
    stdout = getattr(error, "stdout", None)
    if stdout:
      print("Stdout:")
      for line in stdout.strip().splitlines():
        print(f"  {line}")
    stderr = getattr(error, "stderr", None)
    if stderr:
      print("Stderr:")
      for line in stderr.strip().splitlines():
        print(f"  {line}")
    print(f"Error: {error}")

    repo_dir = getattr(self, "nighthawk_git_repo_dir", None)
    if repo_dir:
      print(f"\nRepository directory: {repo_dir}")
    print("The process is paused to give you the opportunity to fix it.")
    print("=" * 78)

    while True:
      choice = (input("\nChoose action: [r]etry step / [p]roceed to next step / [e]xit"
                      " process: ").strip().lower())
      if choice in ("r", "retry"):
        return "retry"
      if choice in ("p", "proceed", "s", "skip", "next"):
        return "proceed"
      if choice in ("e", "exit", "a", "abort", "q", "quit"):
        return "exit"
      print(f"Invalid input '{choice}'. Please enter 'r' (retry), 'p'"
            " (proceed), or 'e' (exit).")

  def _run_step(self, step: TStep) -> None:
    """Dispatch execution of a single step to its @register_step handler.

    Args:
      step: The step enum member to execute.
    """
    handler = self._step_handlers.get(step)
    if handler is None:
      raise ValueError(f"{step} is not supported.")
    handler()

  def _run_steps(self, prior_attempts: int = 0) -> dict[TStep, StepResult]:
    """Run all pending steps in sequence.

    Args:
      prior_attempts: Number of prior LLM agent repair attempts.

    Returns:
      A dictionary mapping steps to their results.
    """
    for step in self.steps:
      if self.step_tracker[step].step_status == StepStatus.SUCCESS:
        logging.info("    %s (cached from checkpoint)", step)
        continue
      if self.step_tracker[step].step_status != StepStatus.PENDING:
        continue
      while True:
        logging.info("    %s", step)
        try:
          self._run_step(step)
          self._set_step_success(step)
          break
        except (
            RuntimeError,
            FileNotFoundError,
            ValueError,
            subprocess.CalledProcessError,
        ) as e:
          self._set_step_failure(step, e)
          if self.agent_invocation and prior_attempts < MAX_AGENT_ATTEMPTS:
            prompt = self._create_agent_prompt(step)
            logging.info("Step %s failed. Invoking agent...", step.name)
            logging.info("Agent prompt:\n\n%s", prompt)
            subprocess.run(
                shlex.split(self.agent_invocation),
                cwd=self.nighthawk_git_repo_dir.parent
                if hasattr(self, "nighthawk_git_repo_dir") else None,
                input=prompt,
                text=True,
                check=True,
            )
            self._set_step_success(step)
            logging.info("Agent fix finished. Proceeding...")
            return self._run_steps(prior_attempts=prior_attempts + 1)

          if self.interactive:
            action = self._prompt_interactive_failure(step, e)
            if action == "retry":
              self._set_step_pending(step)
              logging.info("Retrying step %s...", step.name)
              continue
            if action == "proceed":
              logging.info(
                  "Proceeding to next step after manual intervention in %s...",
                  step.name,
              )
              self._set_step_success(step)
              break
            logging.info("Exiting on user request during step %s.", step.name)
            return self.step_tracker
          return self.step_tracker

    return self.step_tracker


class EnvoyCommitIntegrationStep(enum.Enum):
  """Steps to integrate a specific Envoy commit into Nighthawk."""

  RESET_UNTRACKED_CHANGES = enum.auto()
  SET_NIGHTHAWK_BAZEL_DEP = enum.auto()
  COPY_EXACT_FILES = enum.auto()
  PATCH_SHARED_FILES = enum.auto()
  SYNC_BAZEL_DEPS = enum.auto()
  REFRESH_BAZEL_LOCKFILE = enum.auto()
  BAZEL_UPDATE_REQUIREMENTS = enum.auto()
  BUILD_NIGHTHAWK = enum.auto()
  TEST_NIGHTHAWK = enum.auto()
  FIX_FORMAT = enum.auto()
  UPDATE_CLI_README = enum.auto()
  GIT_ADD_INTEGRATION = enum.auto()


class EnvoyCommitIntegration(StepHandler[EnvoyCommitIntegrationStep]):
  """Handles the steps to integrate a specific Envoy commit into Nighthawk."""

  def __init__(
      self,
      nighthawk_git_repo_dir: pathlib.Path,
      envoy_git_repo_dir: pathlib.Path,
      current_envoy_commit: str,
      target_envoy_commit: str,
      agent_invocation: str | None = None,
      interactive: bool = True,
      checkpoint_store: CheckpointStore | None = None,
  ) -> None:
    """Initialize the EnvoyCommitIntegration.

    Args:
      nighthawk_git_repo_dir: The path to the local Nighthawk git repository.
      envoy_git_repo_dir: The path to the local Envoy git repository.
      current_envoy_commit: The current Envoy commit hash in Nighthawk.
      target_envoy_commit: The target Envoy commit hash to integrate.
      agent_invocation: Command template to invoke an LLM agent to fix step
        failures.
      interactive: Whether to pause and prompt the user when a step fails.
      checkpoint_store: Optional CheckpointStore for persisting completed steps.
    """
    super().__init__(
        EnvoyCommitIntegrationStep,
        agent_invocation=agent_invocation,
        interactive=interactive,
        checkpoint_store=checkpoint_store,
    )
    self.nighthawk_git_repo_dir = nighthawk_git_repo_dir
    self.envoy_git_repo_dir = envoy_git_repo_dir
    self.current_envoy_commit = current_envoy_commit
    self.target_envoy_commit = target_envoy_commit

    if self.checkpoint_store is not None:
      self.checkpoint_store.bind_target(
          base_commit=self.current_envoy_commit,
          target_commit=self.target_envoy_commit,
          working_tree_commit=self._get_working_tree_envoy_commit(),
      )
      self._restore_checkpointed_steps()

  def _step_checkpoint_key(self, step: EnvoyCommitIntegrationStep) -> str:
    """Return the target-scoped checkpoint key for an integration step.

    Args:
      step: The integration step enum member.

    Returns:
      The target-scoped key string.
    """
    return f"step:{self.target_envoy_commit}:{step.name}"

  def _get_working_tree_envoy_commit(self) -> str | None:
    """Read ENVOY_COMMIT from Nighthawk's working-tree MODULE.bazel.

    Returns:
      The 40-char Envoy commit hash if present, else None.
    """
    module_file = self.nighthawk_git_repo_dir / "MODULE.bazel"
    if not module_file.exists():
      return None
    match = _ENVOY_COMMIT_PATTERN.search(module_file.read_text(encoding="utf-8"))
    return match.group(1) if match else None

  def _sync_bazelrc_registry_sha(self, envoy_bazelrc_content: str) -> None:
    """Update the pinned bazel-registry SHA in Nighthawk's .bazelrc.

    Args:
      envoy_bazelrc_content: Content of Envoy's .bazelrc at the target commit.
    """
    nighthawk_bazelrc = self.nighthawk_git_repo_dir / ".bazelrc"
    if not nighthawk_bazelrc.exists():
      return
    match = _BAZEL_REGISTRY_URL_PATTERN.search(envoy_bazelrc_content)
    if not match:
      return
    registry_sha = match.group(2)
    nighthawk_content = nighthawk_bazelrc.read_text(encoding="utf-8")
    updated_content = _BAZEL_REGISTRY_URL_PATTERN.sub(rf"\g<1>{registry_sha}", nighthawk_content)
    if updated_content != nighthawk_content:
      nighthawk_bazelrc.write_text(updated_content, encoding="utf-8")

  def _sync_module_bazel_deps(self) -> None:
    """Sync bazel_dep versions in Nighthawk's MODULE.bazel with Envoy's."""
    nighthawk_module_file = self.nighthawk_git_repo_dir / "MODULE.bazel"
    envoy_module_content = _run_command(
        ["git", "show", f"{self.target_envoy_commit}:MODULE.bazel"],
        cwd=self.envoy_git_repo_dir,
    )
    envoy_deps = {
        match.group(1): match.group(2) for match in _BAZEL_DEP_PATTERN.finditer(envoy_module_content)
    }
    envoy_module_match = _ENVOY_MODULE_VERSION_PATTERN.search(envoy_module_content)
    if envoy_module_match:
      envoy_deps["envoy"] = envoy_module_match.group(1)
    nighthawk_content = nighthawk_module_file.read_text(encoding="utf-8")

    def _replace_dep(match: re.Match[str]) -> str:
      """Replace a bazel_dep version when Envoy specifies a different version."""
      dep_name, nighthawk_version = match.group(1), match.group(2)
      envoy_version = envoy_deps.get(dep_name)
      if envoy_version and nighthawk_version != envoy_version:
        logging.info("%s: %s -> %s", dep_name, nighthawk_version, envoy_version)
        return f'bazel_dep(name = "{dep_name}", version = "{envoy_version}"'
      return match.group(0)

    updated_content = _BAZEL_DEP_PATTERN.sub(_replace_dep, nighthawk_content)
    if updated_content != nighthawk_content:
      nighthawk_module_file.write_text(updated_content, encoding="utf-8")

  def _update_requirements_in_date(self) -> None:
    """Update '# Last updated' in tools/base/requirements.in if requirements.txt changed."""
    diff_result = subprocess.run(
        ["git", "diff", "--quiet", "--", "tools/base/requirements.txt"],
        cwd=self.nighthawk_git_repo_dir,
        check=False,
    )
    if diff_result.returncode == 0:
      return
    requirements_in = self.nighthawk_git_repo_dir / "tools/base/requirements.in"
    if not requirements_in.exists():
      return
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    content = requirements_in.read_text(encoding="utf-8")
    updated = _LAST_UPDATED_PATTERN.sub(f"# Last updated {today}", content, count=1)
    if updated != content:
      requirements_in.write_text(updated, encoding="utf-8")

  def _patch_merge_conflict_instructions(self) -> str:
    """Generate instructions for resolving merge conflicts in shared files.

    Returns:
      A string containing detailed instructions for the user.
    """
    return textwrap.dedent(f"""\
        New merge conflicts from integrating changes in shared files.

        Nighthawk maintains separate copies of a subset of files from the Envoy
        repository: {", ".join(SHARED_FILES)}

        These files are used for configuring the automated build and test
        environments and should be kept in-sync between the two repos. Nighthawk's
        copies contain small modifications necessary for the testing environment
        and marked inline with `# unique`.

        Integrating changes in shared files from commit {self.target_envoy_commit}
        could not be completed cleanly.

        For merge conflicts, inspect Nighthawk's copy and combine the Envoy changes
        with the old Nighthawk version into an updated Nighthawk-specific "# unique" line.
        """)

  @checkpointed(lambda self, shared_file, *_: (f"merge:{self.target_envoy_commit}:{shared_file}"))
  def _apply_shared_file_3way_merge(self, shared_file: str, target_content: str) -> None:
    """Run a 3-way git merge-file on a single shared file once per target commit.

    Args:
      shared_file: Repo-relative path to the shared file.
      target_content: Content of the shared file in Envoy at target_envoy_commit.

    Raises:
      RuntimeError: If git merge-file fails with a fatal exit code.
    """
    nighthawk_file = self.nighthawk_git_repo_dir / shared_file
    with (
        tempfile.NamedTemporaryFile() as base_file,
        tempfile.NamedTemporaryFile() as target_file,
    ):
      base_content = _run_command(
          ["git", "show", f"{self.current_envoy_commit}:{shared_file}"],
          cwd=self.envoy_git_repo_dir,
      )
      if shared_file == ".bazelrc":
        self._sync_bazelrc_registry_sha(base_content)
      base_file.write((base_content + "\n").encode("utf-8"))
      base_file.flush()

      target_file.write((target_content + "\n").encode("utf-8"))
      target_file.flush()

      merge_result = subprocess.run(
          [
              "git",
              "merge-file",
              str(nighthawk_file),
              base_file.name,
              target_file.name,
          ],
          cwd=self.nighthawk_git_repo_dir,
          capture_output=True,
          text=True,
          check=False,
      )
      if merge_result.returncode < 0 or merge_result.returncode > 127:
        raise RuntimeError(f"git merge-file failed for {nighthawk_file} (exit code"
                           f" {merge_result.returncode}): {merge_result.stderr}")

  @register_step(EnvoyCommitIntegrationStep.RESET_UNTRACKED_CHANGES, checkpoint=True)
  def _reset_untracked_changes(self) -> None:
    """Reset any uncommitted or untracked changes in Nighthawk."""
    if self.checkpoint_store is not None:
      self.checkpoint_store.clear_target(self.target_envoy_commit)
    _run_command(["git", "checkout", "."], cwd=self.nighthawk_git_repo_dir)
    _run_command(["git", "clean", "-fd"], cwd=self.nighthawk_git_repo_dir)

  @register_step(EnvoyCommitIntegrationStep.SET_NIGHTHAWK_BAZEL_DEP, checkpoint=True)
  def _set_nighthawk_bazel_dep(self) -> None:
    """Update ENVOY_COMMIT in Nighthawk's MODULE.bazel."""
    module_file = self.nighthawk_git_repo_dir / "MODULE.bazel"
    _run_sed_replace(
        old_pattern='ENVOY_COMMIT = ".*"',
        new_pattern=f'ENVOY_COMMIT = "{self.target_envoy_commit}"',
        filename=str(module_file),
    )
    if self.checkpoint_store is not None:
      self.checkpoint_store.base_envoy_commit = self.current_envoy_commit
      self.checkpoint_store.target_envoy_commit = self.target_envoy_commit
      self.checkpoint_store.save()

  @register_step(EnvoyCommitIntegrationStep.COPY_EXACT_FILES, checkpoint=True)
  def _copy_exact_files(self) -> None:
    """Checkout target_envoy_commit in Envoy and copy COPIED_FILES verbatim."""
    _run_command(
        ["git", "checkout", "--force", self.target_envoy_commit],
        cwd=self.envoy_git_repo_dir,
    )
    for copied_file in COPIED_FILES:
      _run_command([
          "cp",
          f"{self.envoy_git_repo_dir}/{copied_file}",
          f"{self.nighthawk_git_repo_dir}/{copied_file}",
      ])

  @register_step(EnvoyCommitIntegrationStep.PATCH_SHARED_FILES, checkpoint=True)
  def _patch_shared_files(self) -> None:
    """3-way merge SHARED_FILES and verify no conflict markers remain."""
    conflicted_files: list[str] = []
    for shared_file in SHARED_FILES:
      nighthawk_file = self.nighthawk_git_repo_dir / shared_file
      if not nighthawk_file.exists():
        continue

      target_content = _run_command(
          ["git", "show", f"{self.target_envoy_commit}:{shared_file}"],
          cwd=self.envoy_git_repo_dir,
      )
      diff_output = _run_command(
          [
              "git",
              "diff",
              f"{self.current_envoy_commit}..{self.target_envoy_commit}",
              "--",
              shared_file,
          ],
          cwd=self.envoy_git_repo_dir,
      )
      if diff_output:
        self._apply_shared_file_3way_merge(shared_file, target_content)

      if has_conflict_markers(nighthawk_file):
        conflicted_files.append(str(nighthawk_file))
      elif shared_file == ".bazelrc":
        self._sync_bazelrc_registry_sha(target_content)

    if conflicted_files:
      raise RuntimeError(f"{self._patch_merge_conflict_instructions()}\n"
                         "Conflict markers (<<<<<<<, =======, >>>>>>>) placed in: "
                         f"{', '.join(conflicted_files)}.")

  @register_step(EnvoyCommitIntegrationStep.SYNC_BAZEL_DEPS, checkpoint=True)
  def _sync_bazel_deps(self) -> None:
    """Sync bazel_dep versions in Nighthawk's MODULE.bazel."""
    self._sync_module_bazel_deps()

  @register_step(EnvoyCommitIntegrationStep.REFRESH_BAZEL_LOCKFILE, checkpoint=True)
  def _refresh_bazel_lockfile(self) -> None:
    """Refresh MODULE.bazel.lock for the updated MODULE.bazel."""
    _run_command(
        ["bazel", "mod", "deps", "--lockfile_mode=refresh"],
        cwd=self.nighthawk_git_repo_dir,
    )

  @register_step(EnvoyCommitIntegrationStep.BAZEL_UPDATE_REQUIREMENTS, checkpoint=True)
  def _bazel_update_requirements(self) -> None:
    """Re-compile Python requirements.txt and update requirements.in date."""
    _run_command(
        ["./ci/do_ci.sh", "fix_requirements"],
        cwd=self.nighthawk_git_repo_dir,
    )
    self._update_requirements_in_date()

  @register_step(EnvoyCommitIntegrationStep.BUILD_NIGHTHAWK, checkpoint=True)
  def _build_nighthawk(self) -> None:
    """Build Nighthawk binaries."""
    _run_command(["./ci/do_ci.sh", "build"], cwd=self.nighthawk_git_repo_dir)

  @register_step(EnvoyCommitIntegrationStep.TEST_NIGHTHAWK, checkpoint=True)
  def _test_nighthawk(self) -> None:
    """Run Nighthawk test suite."""
    _run_command(["./ci/do_ci.sh", "test"], cwd=self.nighthawk_git_repo_dir)

  @register_step(EnvoyCommitIntegrationStep.FIX_FORMAT, checkpoint=True)
  def _fix_format(self) -> None:
    """Run Nighthawk code formatters."""
    _run_command(["./ci/do_ci.sh", "fix_format"], cwd=self.nighthawk_git_repo_dir)

  @register_step(EnvoyCommitIntegrationStep.UPDATE_CLI_README, checkpoint=True)
  def _update_cli_readme(self) -> None:
    """Regenerate CLI documentation in README.md."""
    _run_command(["./ci/do_ci.sh", "fix_docs"], cwd=self.nighthawk_git_repo_dir)

  @register_step(EnvoyCommitIntegrationStep.GIT_ADD_INTEGRATION, checkpoint=True)
  def _git_add_integration(self) -> None:
    """Stage all integrated changes in the Nighthawk repository."""
    _run_command(["git", "add", "."], cwd=self.nighthawk_git_repo_dir)

  def run_envoy_commit_integration_steps(self,) -> dict[EnvoyCommitIntegrationStep, StepResult]:
    """Run all steps for integrating the target Envoy commit.

    Returns:
      A dictionary mapping EnvoyCommitIntegrationStep to StepResult.
    """
    return self._run_steps()


class NighthawkEnvoyUpdateStep(enum.Enum):
  """Overall steps for updating Envoy dependency in Nighthawk."""

  CHECK_NIGHTHAWK_DIR = enum.auto()
  CHECK_NIGHTHAWK_GIT_REPO = enum.auto()
  CHECK_NIGHTHAWK_GIT_SIGNING = enum.auto()
  CHECK_NIGHTHAWK_GIT_STATUS = enum.auto()
  CHECK_NIGHTHAWK_UPSTREAM_REMOTE = enum.auto()
  CHECK_NIGHTHAWK_ORIGIN_REMOTE = enum.auto()
  SYNC_NIGHTHAWK_REPO = enum.auto()
  CHECKOUT_UPDATE_BRANCH = enum.auto()
  GET_ORIGINAL_ENVOY_COMMIT = enum.auto()
  CLONE_ENVOY = enum.auto()
  GET_ENVOY_COMMIT_RANGE = enum.auto()
  BAZEL_CLEAN_EXPUNGE = enum.auto()
  FIND_LATEST_TRIVIAL_MERGE = enum.auto()
  COMMIT_AND_PUSH_UPDATE_BRANCH = enum.auto()
  APPLY_PARTIAL_INTEGRATION = enum.auto()


class NighthawkEnvoyUpdate(StepHandler[NighthawkEnvoyUpdateStep]):
  """Handles the steps to update the Envoy dependency in Nighthawk."""

  def __init__(
      self,
      nighthawk_git_repo_dir: pathlib.Path,
      branch_name: str,
      envoy_clone_depth: int,
      sync_nighthawk_repo: bool,
      bisect: bool = False,
      interactive: bool = True,
      agent_invocation: str | None = None,
      expunge: bool = False,
      reset_state: bool = False,
      checkpoint_store: CheckpointStore | None = None,
  ) -> None:
    """Initialize the NighthawkEnvoyUpdate.

    Args:
      nighthawk_git_repo_dir: The path to the local Nighthawk git repository.
      branch_name: The name of the branch to create for the update.
      envoy_clone_depth: The depth to use when cloning the Envoy repository.
      sync_nighthawk_repo: Whether to sync the Nighthawk repo with upstream.
      bisect: Whether to bisect Envoy commits to find the latest passing commit.
      interactive: Whether to pause and prompt the user when a step fails.
      agent_invocation: Command template to invoke an LLM agent to fix step
        failures.
      expunge: Whether to run bazel clean --expunge before building.
      reset_state: Whether to discard any existing checkpoint state file.
      checkpoint_store: Optional CheckpointStore instance (defaults to
        .git/nighthawk_envoy_updater_state.json).
    """
    self.nighthawk_git_repo_dir = nighthawk_git_repo_dir.expanduser()
    self.branch_name = branch_name
    self.envoy_clone_depth = envoy_clone_depth
    self.bisect = bisect
    self.expunge = expunge

    # Keep the cached Envoy clone outside nighthawk_git_repo_dir so
    # @envoy//tools/code_format:check_format does not scan Envoy's tree.
    self.envoy_git_repo_dir = (self.nighthawk_git_repo_dir.parent / ".nighthawk-envoy-clone-cache")

    state_path = (self.nighthawk_git_repo_dir / ".git" / "nighthawk_envoy_updater_state.json")
    if checkpoint_store is None:
      checkpoint_store = CheckpointStore.load(state_path)
    if reset_state:
      checkpoint_store.reset()

    super().__init__(
        NighthawkEnvoyUpdateStep,
        agent_invocation=agent_invocation,
        interactive=interactive,
        checkpoint_store=checkpoint_store,
    )

    self.current_envoy_commit: str | None = None
    self.envoy_commits_current_to_latest: list[str] | None = None
    self.best_envoy_commit: str | None = None
    self.first_non_trivial_commit: str | None = None
    self.envoy_commit_integration_results: dict[str, dict[EnvoyCommitIntegrationStep,
                                                          StepResult]] = {}

    if not sync_nighthawk_repo or self._is_resuming_in_progress_target():
      self._set_step_not_planned(NighthawkEnvoyUpdateStep.SYNC_NIGHTHAWK_REPO)
    if not self.expunge:
      self._set_step_not_planned(NighthawkEnvoyUpdateStep.BAZEL_CLEAN_EXPUNGE)

  def _get_working_tree_envoy_commit(self) -> str | None:
    """Return ENVOY_COMMIT from Nighthawk's working-tree MODULE.bazel."""
    module_file = self.nighthawk_git_repo_dir / "MODULE.bazel"
    if not module_file.exists():
      return None
    match = _ENVOY_COMMIT_PATTERN.search(module_file.read_text(encoding="utf-8"))
    return match.group(1) if match else None

  def _get_head_envoy_commit(self) -> str | None:
    """Return ENVOY_COMMIT from Nighthawk's committed HEAD:MODULE.bazel."""
    try:
      head_content = _run_command(
          ["git", "show", "HEAD:MODULE.bazel"],
          cwd=self.nighthawk_git_repo_dir,
      )
    except subprocess.CalledProcessError:
      return None
    match = _ENVOY_COMMIT_PATTERN.search(head_content)
    return match.group(1) if match else None

  def _is_resuming_in_progress_target(self) -> bool:
    """Return True if the working tree has an active checkpointed integration."""
    if (self.checkpoint_store is None or not self.checkpoint_store.target_envoy_commit):
      return False
    return (self._get_working_tree_envoy_commit() == self.checkpoint_store.target_envoy_commit)

  def _get_print_vars(self) -> dict[str, Any]:
    """Build a dictionary of instance variables to print for debugging."""
    print_vars = super()._get_print_vars()
    print_vars.pop("envoy_commits_current_to_latest", None)
    return print_vars

  def _build_commit_message(self, commit: str) -> str:
    """Build a commit message for the Envoy update.

    Args:
      commit: The Envoy commit hash being updated to.

    Returns:
      The formatted commit message string.
    """
    commit_datetime = _run_command(
        [
            "env",
            "TZ=UTC0",
            "git",
            "show",
            "-s",
            "--date=format-local:%Y-%m-%dT%H:%M:%SZ",
            "--pretty=format:%cd",
            commit,
        ],
        cwd=self.envoy_git_repo_dir,
    )
    return (f"Updating Envoy version to {commit[:7]} ({commit_datetime})\n"
            "\n"
            f"See https://github.com/envoyproxy/envoy/commit/{commit}.")

  def _non_trivial_merge_instructions(self) -> str:
    """Generate instructions for the user when a non-trivial merge is required.

    Returns:
      A string containing instructions on how to resolve the merge conflict.
    """
    envoy_commit = self.first_non_trivial_commit
    branch_name = self.branch_name
    results = self.envoy_commit_integration_results.get(envoy_commit, {})

    return textwrap.dedent(f"""\
        Ran partial integration of Envoy commit {envoy_commit}.

        The Nighthawk repository branch {branch_name} has been left in a state
        with the Envoy commit {envoy_commit} integration applied. This integration
        introduced merge conflicts, returned tooling errors, or failed to build
        and pass its tests.

        Please address any merge conflicts, tooling failures, and build or test
        failures.

        Once all errors are resolved, commit your manual changes using:

          git -C {self.nighthawk_git_repo_dir} add . && \\
            git -C {self.nighthawk_git_repo_dir} commit -m '{self._build_commit_message(self.first_non_trivial_commit)}'

        You can then re-run the script if there are further Envoy commits to integrate.

        ### EnvoyCommitIntegrationResults:

        {format_step_results(results)}
        """)

  @register_step(NighthawkEnvoyUpdateStep.CHECK_NIGHTHAWK_DIR)
  def _check_nighthawk_dir(self) -> None:
    """Verify the Nighthawk repository directory exists."""
    if not self.nighthawk_git_repo_dir.is_dir():
      raise RuntimeError(f"Nighthawk directory not found: {self.nighthawk_git_repo_dir}")

  @register_step(NighthawkEnvoyUpdateStep.CHECK_NIGHTHAWK_GIT_REPO)
  def _check_nighthawk_git_repo(self) -> None:
    """Verify the Nighthawk directory is a git work tree."""
    try:
      _run_command(
          ["git", "rev-parse", "--is-inside-work-tree"],
          cwd=self.nighthawk_git_repo_dir,
      )
    except subprocess.CalledProcessError as e:
      raise RuntimeError(f"Nighthawk directory {self.nighthawk_git_repo_dir} is not a git"
                         " repository.") from e

  @register_step(NighthawkEnvoyUpdateStep.CHECK_NIGHTHAWK_GIT_SIGNING)
  def _check_nighthawk_git_signing(self) -> None:
    """Verify the DCO prepare-commit-msg hook is installed."""
    try:
      _run_command(
          [
              "cmp",
              "-s",
              "support/hooks/prepare-commit-msg",
              ".git/hooks/prepare-commit-msg",
          ],
          cwd=self.nighthawk_git_repo_dir,
      )
    except subprocess.CalledProcessError as e:
      raise RuntimeError(f"Nighthawk directory {self.nighthawk_git_repo_dir} is not"
                         " configured to use a signing key with commits.") from e

  @register_step(NighthawkEnvoyUpdateStep.CHECK_NIGHTHAWK_GIT_STATUS)
  def _check_nighthawk_git_status(self) -> None:
    """Verify Nighthawk working tree is clean unless resuming a checkpoint."""
    if not _run_command(["git", "status", "--porcelain"], cwd=self.nighthawk_git_repo_dir):
      return
    if self._is_resuming_in_progress_target():
      logging.info(
          "Resuming checkpointed integration for %s with uncommitted changes.",
          self.checkpoint_store.target_envoy_commit,
      )
      return
    raise RuntimeError("Nighthawk has uncommitted changes. Please reset or commit them.")

  @register_step(NighthawkEnvoyUpdateStep.CHECK_NIGHTHAWK_UPSTREAM_REMOTE)
  def _check_nighthawk_upstream_remote(self) -> None:
    """Verify the upstream remote points to envoyproxy/nighthawk."""
    expected_upstream = "https://github.com/envoyproxy/nighthawk"
    try:
      upstream_url = _run_command(
          ["git", "remote", "get-url", "upstream"],
          cwd=self.nighthawk_git_repo_dir,
      )
      if upstream_url != expected_upstream:
        raise RuntimeError(f"Nighthawk directory {self.nighthawk_git_repo_dir} remote"
                           f" upstream is not set to {expected_upstream}. Got:"
                           f" {upstream_url}")
    except subprocess.CalledProcessError as e:
      raise RuntimeError(f"Nighthawk directory {self.nighthawk_git_repo_dir} does not have"
                         f" a remote named upstream set to {expected_upstream}.") from e

  @register_step(NighthawkEnvoyUpdateStep.CHECK_NIGHTHAWK_ORIGIN_REMOTE)
  def _check_nighthawk_origin_remote(self) -> None:
    """Verify the origin remote is configured."""
    try:
      _run_command(
          ["git", "remote", "get-url", "origin"],
          cwd=self.nighthawk_git_repo_dir,
      )
    except subprocess.CalledProcessError as e:
      raise RuntimeError("Failed to get remote origin URL. Is"
                         f" {self.nighthawk_git_repo_dir} a git repository with an origin"
                         " remote configured?") from e

  @register_step(NighthawkEnvoyUpdateStep.SYNC_NIGHTHAWK_REPO)
  def _sync_nighthawk_repo(self) -> None:
    """Fetch upstream/main and push to origin/main."""
    _run_command(["git", "checkout", "main"], cwd=self.nighthawk_git_repo_dir)
    _run_command(
        ["git fetch --all && git merge upstream/main && git push origin main"],
        cwd=self.nighthawk_git_repo_dir,
        interactive=True,
        shell=True,
    )

  @register_step(NighthawkEnvoyUpdateStep.CHECKOUT_UPDATE_BRANCH)
  def _checkout_update_branch(self) -> None:
    """Create or checkout the target update branch."""
    current_branch = _run_command(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=self.nighthawk_git_repo_dir,
    )
    if (current_branch == self.branch_name and self._is_resuming_in_progress_target()):
      logging.info(
          "Staying on branch %s to resume in-progress integration.",
          self.branch_name,
      )
      return
    try:
      _run_command(
          ["git", "checkout", "-b", self.branch_name, "origin/main"],
          cwd=self.nighthawk_git_repo_dir,
      )
    except subprocess.CalledProcessError:
      _run_command(
          ["git", "checkout", self.branch_name],
          cwd=self.nighthawk_git_repo_dir,
      )
      _run_command(["git", "rebase", "origin/main"], cwd=self.nighthawk_git_repo_dir)

  @register_step(NighthawkEnvoyUpdateStep.GET_ORIGINAL_ENVOY_COMMIT)
  def _get_original_envoy_commit(self) -> None:
    """Extract the base ENVOY_COMMIT from HEAD:MODULE.bazel (or working tree)."""
    self.current_envoy_commit = (self._get_head_envoy_commit() or
                                 self._get_working_tree_envoy_commit())
    if not self.current_envoy_commit:
      raise RuntimeError("Failed to extract current Envoy commit from"
                         f" {self.nighthawk_git_repo_dir / 'MODULE.bazel'}")

  @register_step(NighthawkEnvoyUpdateStep.CLONE_ENVOY)
  def _clone_envoy(self) -> None:
    """Clone Envoy or fetch latest commits into the cached Envoy clone."""
    if (self.envoy_git_repo_dir / ".git").is_dir():
      _run_command(
          [
              "git",
              "fetch",
              f"--depth={self.envoy_clone_depth}",
              "origin",
              "main",
          ],
          cwd=self.envoy_git_repo_dir,
      )
      return
    if self.envoy_git_repo_dir.exists():
      shutil.rmtree(self.envoy_git_repo_dir)
    _run_command([
        "git",
        "clone",
        f"--depth={self.envoy_clone_depth}",
        "https://github.com/envoyproxy/envoy.git",
        str(self.envoy_git_repo_dir),
    ])

  @register_step(NighthawkEnvoyUpdateStep.GET_ENVOY_COMMIT_RANGE)
  def _get_envoy_commit_range(self) -> None:
    """Resolve the list of Envoy commits between current_envoy_commit and origin/main."""
    try:
      current_to_latest_raw = _run_command(
          [
              "git",
              "log",
              "--reverse",
              "--pretty=%H",
              f"{self.current_envoy_commit}..origin/main",
          ],
          cwd=self.envoy_git_repo_dir,
      )
    except subprocess.CalledProcessError as e:
      raise RuntimeError(f"The current Envoy commit {self.current_envoy_commit} used in"
                         " Nighthawk was not found in the cloned Envoy repository"
                         " history. Please try again with a larger --envoy_clone_depth"
                         f" value than {self.envoy_clone_depth}.") from e

    self.envoy_commits_current_to_latest = [c for c in current_to_latest_raw.splitlines() if c]
    if not self.envoy_commits_current_to_latest:
      logging.info("Nighthawk is up-to-date, no new commits in Envoy repo.")
      sys.exit(0)
    if not self.bisect:
      self.envoy_commits_current_to_latest = [self.envoy_commits_current_to_latest[-1]]

  @register_step(NighthawkEnvoyUpdateStep.BAZEL_CLEAN_EXPUNGE)
  def _bazel_clean_expunge(self) -> None:
    """Run bazel clean --expunge when explicitly requested via --expunge."""
    _run_command(["bazel", "clean", "--expunge"], cwd=self.nighthawk_git_repo_dir)

  @checkpointed(lambda self, current_commit, target_commit:
                (f"bisect:{current_commit}..{target_commit}"))
  def _evaluate_commit_integration(self, current_commit: str, target_commit: str) -> bool:
    """Run EnvoyCommitIntegration and return whether it integrated cleanly.

    Args:
      current_commit: The base Envoy commit hash.
      target_commit: The candidate Envoy commit hash to test.

    Returns:
      True if all integration steps succeeded.
    """
    results = EnvoyCommitIntegration(
        nighthawk_git_repo_dir=self.nighthawk_git_repo_dir,
        envoy_git_repo_dir=self.envoy_git_repo_dir,
        current_envoy_commit=current_commit,
        target_envoy_commit=target_commit,
        agent_invocation=self.agent_invocation,
        interactive=self.interactive,
        checkpoint_store=self.checkpoint_store,
    ).run_envoy_commit_integration_steps()
    self.envoy_commit_integration_results[target_commit] = results
    return is_clean_results(results)

  @register_step(NighthawkEnvoyUpdateStep.FIND_LATEST_TRIVIAL_MERGE)
  def _find_latest_trivial_merge(self) -> None:
    """Integrate the latest Envoy commit or bisect to find the latest clean commit."""
    if not self.bisect:
      target_envoy_commit = self.envoy_commits_current_to_latest[-1]
      logging.info(
          "Attempting to integrate latest Envoy commit: %s"
          " (https://github.com/envoyproxy/envoy/commit/%s)",
          target_envoy_commit,
          target_envoy_commit,
      )
      results = EnvoyCommitIntegration(
          nighthawk_git_repo_dir=self.nighthawk_git_repo_dir,
          envoy_git_repo_dir=self.envoy_git_repo_dir,
          current_envoy_commit=self.current_envoy_commit,
          target_envoy_commit=target_envoy_commit,
          agent_invocation=self.agent_invocation,
          interactive=self.interactive,
          checkpoint_store=self.checkpoint_store,
      ).run_envoy_commit_integration_steps()
      self.envoy_commit_integration_results[target_envoy_commit] = results

      if is_clean_results(results):
        logging.info("Successfully integrated latest Envoy commit.")
        self.best_envoy_commit = target_envoy_commit
        self.first_non_trivial_commit = None
        self._set_step_not_planned(NighthawkEnvoyUpdateStep.APPLY_PARTIAL_INTEGRATION)
      else:
        logging.info("Integration of latest Envoy commit failed.")
        self.best_envoy_commit = None
        self.first_non_trivial_commit = target_envoy_commit
        self._set_step_not_planned(NighthawkEnvoyUpdateStep.COMMIT_AND_PUSH_UPDATE_BRANCH)
      return

    current_envoy_commit = self.current_envoy_commit
    low = 0
    high = len(self.envoy_commits_current_to_latest) - 1
    index_to_test = high

    while low <= high:
      target_envoy_commit = self.envoy_commits_current_to_latest[index_to_test]
      logging.info("Bisection status:")
      for i, commit in enumerate(self.envoy_commits_current_to_latest):
        status = None
        results = self.envoy_commit_integration_results.get(commit)
        if results:
          status = "PASSED" if is_clean_results(results) else "FAILED"
        elif target_envoy_commit == commit:
          status = "---->"
        elif i in (0, len(self.envoy_commits_current_to_latest) - 1):
          status = " "
        if status:
          logging.info(
              "[%s] https://github.com/envoyproxy/envoy/commit/%s",
              f"{status:^8}",
              commit,
          )

      clean_integration = self._evaluate_commit_integration(current_envoy_commit,
                                                            target_envoy_commit)
      if clean_integration:
        current_envoy_commit = target_envoy_commit
        self.best_envoy_commit = target_envoy_commit
        low = index_to_test + 1
      else:
        self.first_non_trivial_commit = target_envoy_commit
        high = index_to_test - 1

      index_to_test = low + (high - low) // 2

    if not self.best_envoy_commit:
      logging.info("Bisecting failed to find an Envoy commit that can be trivially"
                   " integrated.")
      self.best_envoy_commit = None
      self.first_non_trivial_commit = self.envoy_commits_current_to_latest[0]
      self._set_step_not_planned(NighthawkEnvoyUpdateStep.COMMIT_AND_PUSH_UPDATE_BRANCH)
    elif self.best_envoy_commit == self.envoy_commits_current_to_latest[-1]:
      logging.info("The latest Envoy commit can be trivially integrated.")
      self.best_envoy_commit = self.envoy_commits_current_to_latest[-1]
      self.first_non_trivial_commit = None
      self._set_step_not_planned(NighthawkEnvoyUpdateStep.APPLY_PARTIAL_INTEGRATION)
    else:
      logging.info("A trivially integrated Envoy commit was found and there are"
                   " further Envoy commits after it.")

  @register_step(NighthawkEnvoyUpdateStep.COMMIT_AND_PUSH_UPDATE_BRANCH)
  def _commit_and_push_update_branch(self) -> None:
    """Commit the integrated Envoy update and push the branch to origin."""
    if not self.best_envoy_commit:
      raise RuntimeError("Nighthawk repo attempting to commit when no trivial Envoy"
                         " merges were found.")
    _run_command(
        [
            "git",
            "commit",
            "-m",
            self._build_commit_message(self.best_envoy_commit),
        ],
        cwd=self.nighthawk_git_repo_dir,
        interactive=True,
    )
    _run_command(
        [
            "git",
            "push",
            "--force",
            "--set-upstream",
            "origin",
            self.branch_name,
        ],
        cwd=self.nighthawk_git_repo_dir,
        interactive=True,
    )
    if self.checkpoint_store is not None:
      self.checkpoint_store.reset()

  @register_step(NighthawkEnvoyUpdateStep.APPLY_PARTIAL_INTEGRATION)
  def _apply_partial_integration(self) -> None:
    """Raise instructions for manual resolution when integration fails."""
    if not self.first_non_trivial_commit:
      raise RuntimeError("Nighthawk repo attempting to apply partial merge when"
                         " first_non_trivial_commit is not set.")
    raise RuntimeError(self._non_trivial_merge_instructions())

  def run_update(self) -> None:
    """Run all steps for the Nighthawk Envoy update process."""
    results = self._run_steps()
    if not is_clean_results(results):
      raise RuntimeError("Failed to update Nighthawk's Envoy commit to the latest Envoy"
                         f" commit:\n\n{format_step_results(results)}")

  def cleanup(self) -> None:
    """Clean up state modifications made by NighthawkEnvoyUpdate."""


def main() -> None:
  """Update Envoy version in Nighthawk."""
  logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
  parser = argparse.ArgumentParser(description="Update Envoy version in Nighthawk.")
  parser.add_argument(
      "--nighthawk_dir",
      type=pathlib.Path,
      default="~/github/nighthawk",
      help="The path to the local Nighthawk git repository clone.",
  )
  parser.add_argument(
      "--envoy_clone_depth",
      type=int,
      default=200,
      help="The depth to use when cloning the Envoy repository.",
  )
  parser.add_argument(
      "--branch_name",
      type=str,
      default=f"update-envoy-{datetime.datetime.now().strftime('%Y%m%d')}",
      help=("The name of the branch to create in the Nighthawk repository for the"
            " update."),
  )
  parser.add_argument(
      "--no_sync_nighthawk_repo",
      action="store_false",
      dest="sync_nighthawk_repo",
      help=("If set, the script will not sync the local Nighthawk repository with"
            " the upstream remote before starting the update process."),
  )
  parser.add_argument(
      "--bisect",
      action="store_true",
      default=False,
      dest="bisect",
      help=("If set, the script will bisect Envoy commits to find the latest"
            " passing commit. By default, bisection is disabled and only the"
            " latest Envoy commit is attempted."),
  )
  parser.add_argument(
      "--skip_bisection",
      action="store_false",
      dest="bisect",
      help=argparse.SUPPRESS,
  )
  parser.add_argument(
      "--interactive",
      action="store_true",
      default=True,
      dest="interactive",
      help=("Enable interactive mode on step failures (pause, fix,"
            " retry/proceed/exit). Enabled by default."),
  )
  parser.add_argument(
      "--no_interactive",
      action="store_false",
      dest="interactive",
      help="Disable interactive mode on step failures.",
  )
  parser.add_argument(
      "--agent_invocation",
      type=str,
      default=None,
      help=("Command to invoke an LLM agent, e.g. 'gemini', with prompt by stdin."),
  )
  parser.add_argument(
      "--expunge",
      action="store_true",
      default=False,
      help="If set, run 'bazel clean --expunge' before integrating.",
  )
  parser.add_argument(
      "--reset_state",
      action="store_true",
      default=False,
      help="If set, discard any saved checkpoint state before running.",
  )

  args = parser.parse_args()

  updater = NighthawkEnvoyUpdate(
      nighthawk_git_repo_dir=args.nighthawk_dir,
      branch_name=args.branch_name,
      envoy_clone_depth=args.envoy_clone_depth,
      sync_nighthawk_repo=args.sync_nighthawk_repo,
      bisect=args.bisect,
      interactive=args.interactive,
      agent_invocation=args.agent_invocation,
      expunge=args.expunge,
      reset_state=args.reset_state,
  )
  try:
    updater.run_update()
  finally:
    updater.cleanup()


if __name__ == "__main__":
  main()
