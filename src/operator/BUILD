package(default_visibility = ["//visibility:public"])
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "operator",
    srcs = [
        "operator.py",
    ],
)

py_library(
    name = "common",
    srcs = [
        "common.py",
    ],
    deps = [
        ":operator",
        "//src:utils",
    ],
)

py_library(
    name = "activation",
    srcs = [
        "activation.py",
    ],
    deps = [
        ":operator"
    ],
)

py_library(
    name = "normalizer",
    srcs = [
        "normalizer.py",
    ],
    deps = [
        ":operator",
    ],
)

py_library(
    name = "loss",
    srcs = [
        "loss.py",
    ],
    deps = [
        ":operator",
        "//src:utils",
    ],
)
