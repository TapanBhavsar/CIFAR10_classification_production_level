from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class Model(ABC):
    def __init__(
        self,
        train_data,
        train_labels,
        test_data,
        test_labels,
        validation_data=None,
        validation_labels=None,
    ):
        self.__train_data = train_data
        self.__train_labels = train_labels
        self.__validation_data = validation_data
        self.__validation_labels = validation_labels
        self.__test_data = test_data
        self.__test_labels = test_labels
        self._input_data = None
        self._input_labels = None

    def initialize_placeholders(
        self,
        input_data_shape=[None, 32, 32, 3],
        input_datatype=tf.float32,
        labels_shape=[None, 10],
        labels_datatype=tf.int32,
    ):
        self._input_data = tf.placeholder(input_datatype, input_data_shape)
        self._input_labels = tf.placeholder(labels_datatype, labels_shape)

    @staticmethod
    def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        # shuffle is used in train the data
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx : start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    @abstractmethod
    def initialize_weights(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass
