from model import Model


class Cifar10CNNModel(Model):
    def __init__(self, train_data, train_labels, test_data, test_labels):
        super(Cifar10CNNModel, self).__init__(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
        )
