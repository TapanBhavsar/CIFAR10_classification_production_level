py_library(
    name = "CIFAR10_serving_lib",
    srcs = glob(["lib/*.py"]),
    imports = ["lib"],
)

py_binary(
    name = "CIFAR10_serving",
    srcs = ["main_engine/CIFAR_10_model_trainer.py"],
    main = "main_engine/CIFAR_10_model_trainer.py",
    deps = [":CIFAR10_serving_lib"],
)
