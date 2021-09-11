package(default_visibility = ["//visibility:public"])
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "jupyter",
    srcs = [
        "jupyter.py",
    ],
    deps = [
        "//src:nn_src",
        "//data/MNIST:mnist_csv_loader"
    ],
)
