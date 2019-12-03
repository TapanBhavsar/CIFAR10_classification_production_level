from model_trainer import ModelTrainer
from CIFAR10_model import CIFAR10Model


class CIFAR10ModelTrainer(ModelTrainer):
    def __init__(
        self, train_data, train_labels, validation_data, validation_labels, test_data, test_labels,
    ):
        super(CIFAR10ModelTrainer, self).__init__(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            validation_data=validation_data,
            validation_labels=validation_labels,
        )

        input_data_shape = [None]
        input_label_shape = [None]
        input_data_shape.extend(train_data.shape[1:])
        input_label_shape.extend(train_labels.shape[1:])

        self._initialize_placeholders(input_data_shape=input_data_shape, labels_shape=input_label_shape)
        self.__CIFAR10_model = CIFAR10Model()

    def train_model(self):
        self.__CIFAR10_model.build_model(self._input_data)

    def save_model(self):
        pass

