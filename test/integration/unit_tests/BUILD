load("@nh_pip3//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

licenses(["notice"])  # Apache 2

py_test(
    name = "test_nighthawk_test_server",
    srcs = ["test_nighthawk_test_server.py"],
    deps = [
        "//test/integration:integration_test_base_lean",
    ],
)

py_test(
    name = "test_utility",
    srcs = ["test_utility.py"],
    deps = [
        "//test/integration:utility_lib",
        "@rules_python//python/runfiles",
        requirement("pytest"),
        requirement("pyyaml"),
    ],
)

py_test(
    name = "test_subprocess_mixin",
    srcs = ["test_subprocess_mixin.py"],
    deps = [
        "//test/integration:integration_test_base_lean",
    ],
)
