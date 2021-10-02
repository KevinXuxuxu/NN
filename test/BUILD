package(default_visibility = ["//visibility:public"])
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "layer_test",
    srcs = ["layer_test.py"],
    deps = [
        "//src:nn_src",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    deps = [
        "//src:nn_src",
    ],
)

py_test(
    name = "operator_test",
    srcs = ["operator_test.py"],
    deps = [
        "//src/operator:activation",
        "//src/operator:common",
        "//src/operator:loss",
        "//src/operator:operator",
    ],
)
