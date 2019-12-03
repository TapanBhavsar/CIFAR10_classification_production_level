from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    def __init__(self):
        self._weights = None
        self._biases = None

    @abstractmethod
    def initialize_weights(self):
        pass

    @abstractmethod
    def create_model(self, input_data_placeholder):
        pass

    @abstractmethod
    def build_model(self, input_data_placeholder):
        pass
