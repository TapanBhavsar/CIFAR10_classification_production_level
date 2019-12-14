from model_prediction import ModelPrediction


def main():
    MODEL_PATH = "./CIFAR10_Serving/model_weights/model.ckpt.meta"
    IMAGE_PATH = "./CIFAR10_Serving/test/test.jpeg"
    model_predictor = ModelPrediction(MODEL_PATH)
    model_predictor.predict_input_image(IMAGE_PATH)


if __name__ == "__main__":
    main()