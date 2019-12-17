import tensorflow as tf
import os
import cv2
import numpy as np

from data_preprocessor import DataPreprocessor


class ModelPrediction:
    LABELS_FILE_PATH = "./CIFAR10_Serving/data/class_names.txt"

    def __init__(self, model_path, parameter_file):
        self._model_path = model_path
        self._parameter_file = parameter_file
        self._session = tf.Session()
        self._input_node = None
        self._prediction_node = None
        self._load_model()
        self._input_shape = self._input_node.shape[1:]

    def is_model_present(self):
        return os.path.exists(self._model_path)

    def _get_class_prediction(self, prediction_array):
        max_value_index = np.argmax(prediction_array)
        with open(self.LABELS_FILE_PATH) as file:
            file_lines = file.readlines()
            return file_lines[max_value_index]

    def _read_image(self, image_path):
        input_image = cv2.imread(image_path)
        input_image = cv2.resize(input_image, (self._input_shape[0], self._input_shape[1]))
        input_image = np.expand_dims(input_image, axis=0)
        preprocessor_prediction = DataPreprocessor(input_image)
        preprocessor_prediction.restore_preprocessing_parameters(file_name=self._parameter_file)
        input_image = preprocessor_prediction.get_reprocessed_data()
        return input_image

    @staticmethod
    def display_present_node_list(graph):
        nodes = [n.name for n in graph.as_graph_def().node]
        print(nodes)

    def _load_model(self, input_tensor_name="inputs:0", output_tensor_name="output_softmax:0"):
        folder_path, _ = os.path.split(self._model_path)
        new_saver = tf.train.import_meta_graph(self._model_path)
        new_saver.restore(self._session, tf.train.latest_checkpoint(folder_path))

        graph = tf.get_default_graph()
        self._input_node = graph.get_tensor_by_name(input_tensor_name)
        self._prediction_node = graph.get_tensor_by_name(output_tensor_name)

    def predict_input_image(self, image_path):
        input_image = self._read_image(image_path)
        output_prediction = self._session.run([self._prediction_node], feed_dict={self._input_node: input_image})[0]
        return self._get_class_prediction(prediction_array=output_prediction), np.max(output_prediction)
