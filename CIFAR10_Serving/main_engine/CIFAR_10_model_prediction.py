from model_prediction import ModelPrediction


def main():
    MODEL_PATH = "../../CIFAR10_serving_trainer.runfiles/CIFAR_10_SERVING/CIFAR10_Serving/model_weights/model.ckpt.meta"
    IMAGE_PATH = "./CIFAR10_Serving/test/test.jpeg"
    PARAMETER_PATH = "../../CIFAR10_serving_trainer.runfiles/CIFAR_10_SERVING/CIFAR10_Serving/parameters/parameters.npy"
    model_predictor = ModelPrediction(MODEL_PATH, PARAMETER_PATH)

    predicted_class = ""
    if model_predictor.is_model_present():
        predicted_class, _ = model_predictor.predict_input_image(IMAGE_PATH)
    else:
        print("please train model first...")

    print("Predicted class: ", predicted_class)


if __name__ == "__main__":
    main()
