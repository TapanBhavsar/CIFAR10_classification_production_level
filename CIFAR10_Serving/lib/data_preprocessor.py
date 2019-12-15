import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import utilities


class DataPreprocessor(object):
    def __init__(self, data, labels=None):
        """ Input data require as numpy formats"""
        self.__data = data
        self.__labels = labels
        self.__mean = None
        self.__max = None
        self.__min = None

    def _calculate_preprocessing_parameters(self, axis=None):
        self.__mean = np.mean(self.__data, axis=axis)
        self.__max = np.max(self.__data)
        self.__min = np.min(self.__data)

    def _standardize_data(self):
        self.__data = (self.__data - self.__mean) / (self.__max - self.__min)

    def one_hot_encode_labels(self, datatype=np.float):
        encoder = OneHotEncoder(dtype=datatype)
        return encoder.fit_transform(self.__labels.reshape(len(self.__labels), -1)).toarray()

    def get_reprocessed_data(self):
        self._standardize_data()
        return self.__data

    def store_preprocessing_parameters(self, file_name="parameters.npy"):
        folder, file = os.path.split(file_name)
        utilities.create_folder(folder)
        self._calculate_preprocessing_parameters()
        store_data = np.asarray([self.__mean, self.__max, self.__min])
        np.save(file_name, store_data)

    def restore_preprocessing_parameters(self, file_name="parameters.npy"):
        load_data = np.load(file_name)
        self.__mean = load_data[0]
        self.__max = load_data[1]
        self.__min = load_data[2]
