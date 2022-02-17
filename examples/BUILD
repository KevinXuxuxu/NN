load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "jupyter",
    srcs = [
        "jupyter.py",
    ],
    deps = [
        "//data/MNIST:mnist_csv_loader",
        "//src:nn_src",
        "//src/operator",
        "//src/operator:activation",
        "//src/operator:common",
        "//src/operator:loss",
        "//src/operator:normalizer",
    ],
)
