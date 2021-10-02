package(default_visibility = ["//visibility:public"])
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "jupyter",
    srcs = [
        "jupyter.py",
    ],
    deps = [
        "//src/operator:activation",
        "//src/operator:common",
        "//src/operator:loss",
        "//src/operator:normalizer",
        "//src/operator:operator",
        "//src:nn_src",
        "//data/MNIST:mnist_csv_loader"
    ],
)
