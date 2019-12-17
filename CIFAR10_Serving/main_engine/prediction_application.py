from flask import Flask, jsonify, request
import os

from model_prediction import ModelPrediction


MODEL_PATH = "../../CIFAR10_serving_trainer.runfiles/CIFAR_10_SERVING/CIFAR10_Serving/model_weights/model.ckpt.meta"
PARAMETER_PATH = "../../CIFAR10_serving_trainer.runfiles/CIFAR_10_SERVING/CIFAR10_Serving/parameters/parameters.npy"
model_predictor = ModelPrediction(MODEL_PATH, PARAMETER_PATH)

UPLOAD_FOLDER = "./CIFAR10_Serving/test"
app = Flask(__name__)


@app.route("/predict_image", methods=["POST"])
def predict_image():
    json_data = request.get_json()
    if json_data["file"] is None:
        return jsonify({"error": "no file"}), 400
    # Image info
    img_file = json_data.get("file")
    # Write image to static directory and do the hot dog check
    image_path = os.path.join(UPLOAD_FOLDER, img_file)
    # img_file.save(image_path)
    if model_predictor.is_model_present():
        predicted_class, predicted_confidence = model_predictor.predict_input_image(image_path)
        response = {"class_name": predicted_class, "confidence": str(predicted_confidence)}
    else:
        response = {"message": "Trained model is not present please train the model first"}
    return jsonify(response), 201


app.run(host="127.0.0.1", port=4000)
