from model_trainer import ModelTrainer
from CIFAR10_model import CIFAR10Model
import tensorflow as tf
import time
import os
import utilities


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
        self.__session = tf.Session()

        self.__initialize_variables = None
        self.__model_saver = None

    def train_model(self, epochs, batch_size):
        model_output, model_softmax_output = self.__CIFAR10_model.build_model(self._input_data)
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_output, labels=self._input_labels))

        optm = tf.train.AdamOptimizer(learning_rate=0.01).minimize(error)
        corr = tf.equal(tf.argmax(model_output, 1), tf.argmax(self._input_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))

        self.__initialize_variables = tf.global_variables_initializer()
        self.__session.run(self.__initialize_variables)
        self.__model_saver = tf.train.Saver()

        for epoch in range(epochs):
            train_err = 0
            train_acc = 0
            train_batches = 0
            start_time = time.time()
            # devide data into mini batch
            for batch in self._iterate_minibatches(self._train_data, self._train_labels, batch_size, shuffle=True):
                inputs, targets = batch
                # this is update weights
                self.__session.run([optm], feed_dict={self._input_data: inputs, self._input_labels: targets})
                # cost function
                err, acc = self.__session.run(
                    [error, accuracy], feed_dict={self._input_data: inputs, self._input_labels: targets}
                )
                train_err += err
                train_acc += acc
                train_batches += 1

            val_err = 0
            val_acc = 0
            val_batches = 0
            # divide validation data into mini batch without shuffle
            for batch in self._iterate_minibatches(
                self._validation_data, self._validation_labels, batch_size, shuffle=False
            ):
                inputs, targets = batch
                # this is update weights
                self.__session.run([optm], feed_dict={self._input_data: inputs, self._input_labels: targets})
                # cost function
                err, acc = self.__session.run(
                    [error, accuracy], feed_dict={self._input_data: inputs, self._input_labels: targets}
                )
                val_err += err
                val_acc += acc
                val_batches += 1
            # print present epoch with total number of epoch
            # print training and validation loss with accuracy
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in self._iterate_minibatches(self._test_data, self._test_labels, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = self.__session.run(
                [error, accuracy], feed_dict={self._input_data: inputs, self._input_labels: targets}
            )  # apply tensor function
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

    def save_model(self, model_path):
        model_folder, model_file = os.path.split(model_path)
        utilities.create_folder(model_folder)
        save_path = self.__model_saver.save(self.__session, model_path)

