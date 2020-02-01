from flask import Flask, jsonify, request
import os
import cv2
import numpy as np

from model_prediction import ModelPrediction


MODEL_PATH = "../../CIFAR10_serving_trainer.runfiles/CIFAR_10_SERVING/CIFAR10_Serving/model_weights/model.ckpt.meta"
PARAMETER_PATH = "../../CIFAR10_serving_trainer.runfiles/CIFAR_10_SERVING/CIFAR10_Serving/parameters/parameters.npy"
model_predictor = ModelPrediction(MODEL_PATH, PARAMETER_PATH)

UPLOAD_FOLDER = "./CIFAR10_Serving/test"
app = Flask(__name__)


@app.route("/predict_image", methods=["POST"])
def predict_image():
    data = request.files["file"]
    encoded_image = np.fromfile(data, np.uint8)
    input_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if model_predictor.is_model_present():
        predicted_class, predicted_confidence = model_predictor.predict_input_image_api(input_image)
        response = {"class_name": predicted_class, "confidence": str(predicted_confidence)}
    else:
        response = {"message": "Trained model is not present please train the model first"}
    return jsonify(response), 201


app.run(host="127.0.0.1", port=4000)

# curl -F "file=@<file path>" http://<localhost:port>/predict_image
