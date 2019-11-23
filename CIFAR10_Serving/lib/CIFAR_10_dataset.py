import pickle
import glob
import numpy
import logging


class CIFAR10Dataset(object):
    def __init__(self, path):
        self.path = path

    @staticmethod
    def format_CIFAR10_dataset(data):
        data = numpy.dstack((data[:, :1024], data[:, 1024:2048], data[:, 2048:]))
        return data.reshape((data.shape[0], 32, 32, 3))

    def read_CIFAR10_dataset(self, batch="data"):
        raw_data = []
        labels = []
        for path in glob.glob("{}*/{}*".format(self.path, batch)):
            with open(path, "rb") as f:
                batch = pickle.load(f, encoding="latin1")
            raw_data.append(batch["data"])
            labels.append(batch["labels"])

        return (
            self.format_CIFAR10_dataset((numpy.concatenate(raw_data))),
            numpy.concatenate(labels).astype(numpy.int32),
        )

    def create_dataset_format(self):
        train_data, train_labels = self.read_CIFAR10_dataset(batch="data")
        test_data, test_labels = self.read_CIFAR10_dataset(batch="test")

        return train_data, train_labels, test_data, test_labels
