py_library(
    name = "CIFAR10_serving_lib",
    srcs = glob(["lib/*.py"]),
    imports = ["lib"],
)

py_binary(
    name = "CIFAR10_serving_trainer",
    srcs = ["main_engine/CIFAR_10_model_trainer.py"],
    main = "main_engine/CIFAR_10_model_trainer.py",
    deps = [":CIFAR10_serving_lib"],
)

py_binary(
    name = "CIFAR10_serving_prediction",
    srcs = ["main_engine/CIFAR_10_model_prediction.py"],
    data = glob([
        "test/*",
        "data/*",
    ]),
    main = "main_engine/CIFAR_10_model_prediction.py",
    deps = [":CIFAR10_serving_lib"],
)
