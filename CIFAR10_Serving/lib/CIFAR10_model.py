from model import Model
import tensorflow as tf


class CIFAR10Model(Model):
    def __init__(self):
        super(Model, self).__init__()
        self.__STANDARD_DEVIATION = 0.01

    def initialize_weights(self):
        self._weights = {
            "w1": tf.Variable(tf.random_normal([5, 5, 3, 16], stddev=self.__STANDARD_DEVIATION)),
            "w2": tf.Variable(tf.random_normal([5, 5, 16, 32], stddev=self.__STANDARD_DEVIATION)),
            "w3": tf.Variable(tf.random_normal([4, 4, 32, 16], stddev=self.__STANDARD_DEVIATION)),
            "w4": tf.Variable(tf.random_normal([4 * 4 * 16, 32], stddev=self.__STANDARD_DEVIATION)),
            "w5": tf.Variable(tf.random_normal([32, 10], stddev=self.__STANDARD_DEVIATION)),
        }

        self._biases = {
            "b1": tf.Variable(tf.random_normal([16], stddev=self.__STANDARD_DEVIATION)),
            "b2": tf.Variable(tf.random_normal([32], stddev=self.__STANDARD_DEVIATION)),
            "b3": tf.Variable(tf.random_normal([16], stddev=self.__STANDARD_DEVIATION)),
            "b4": tf.Variable(tf.random_normal([32], stddev=self.__STANDARD_DEVIATION)),
            "b5": tf.Variable(tf.random_normal([10], stddev=self.__STANDARD_DEVIATION)),
        }

    def create_model(self, input_data_placeholder):
        conv1 = tf.nn.conv2d(input_data_placeholder, self._weights["w1"], strides=[1, 1, 1, 1], padding="SAME")
        conv1 = tf.nn.bias_add(conv1, self._biases["b1"])
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        conv2 = tf.nn.conv2d(pool1, self._weights["w2"], strides=[1, 1, 1, 1], padding="SAME")
        conv2 = tf.nn.bias_add(conv2, self._biases["b2"])
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        conv3 = tf.nn.conv2d(pool2, self._weights["w3"], strides=[1, 1, 1, 1], padding="SAME")
        conv3 = tf.nn.bias_add(conv3, self._biases["b3"])
        conv3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        shape = pool3.get_shape().as_list()
        dense = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])
        dense1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dense, self._weights["w4"]), self._biases["b4"]))

        # used for training the CNN model
        out = tf.nn.bias_add(tf.matmul(dense1, self._weights["w5"]), self._biases["b5"], name="output_model")

        # used after training the CNN
        softmax = tf.nn.softmax(out, name="output_softmax")
        return out, softmax

    def build_model(self, input_data_placeholder):
        self.initialize_weights()
        return self.create_model(input_data_placeholder=input_data_placeholder)
