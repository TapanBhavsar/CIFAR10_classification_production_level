import numpy as np
from sklearn.preprocessing import OneHotEncoder


class DataPreprocessor(object):
    def __init__(self, data, labels):
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
        return encoder.fit_transform(
            self.__labels.reshape(len(self.__labels), -1)
        ).toarray()

    def get_reprocessed_data(self):
        self._standardize_data()
        return self.__data

    def store_preprocessing_parameters(self):
        self._calculate_preprocessing_parameters()

    def restore_preprocessing_parameters(self):
        pass
