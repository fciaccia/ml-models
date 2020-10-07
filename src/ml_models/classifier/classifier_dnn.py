import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, model_from_json

# Disable annoying tensorflow warning messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


DEFAULT_N_HIDDEN_LAYERS = 2
DEFAULT_ALPHA = 3
DEFAULT_OPTIMIZER = 'adam'
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'
DEFAULT_TRAINING_METRICS = ['accuracy']
DEFAULT_TRAINING_EPOCHS = 50
DEFAULT_BATCH_SIZE = 100
DEFAULT_VERBOSITY = 0


class ClassifierDNN():

    def __init__(self, input_dim, output_dim,
                 n_hidden_layers=DEFAULT_N_HIDDEN_LAYERS,
                 alpha=DEFAULT_ALPHA,
                 optimizer=DEFAULT_OPTIMIZER,
                 metrics=DEFAULT_TRAINING_METRICS,
                 epochs=DEFAULT_TRAINING_EPOCHS,
                 batch_size=DEFAULT_BATCH_SIZE,
                 verbosity=DEFAULT_VERBOSITY,
                 loss_function=None,
                 do_plots=False,
                 **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.alpha = alpha
        self.optimizer = optimizer
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbosity = verbosity
        self.do_plots = do_plots
        if loss_function is None:
            if self.output_dim == 1:
                self.loss_function = 'mse'
            else:
                self.loss_function = 'sparse_categorical_crossentropy'
        self._clf = None

    def train(self, data, labels, test_size):
        self._build_neural_network(len(data))
        history = self._clf.fit(data, labels, validation_split=test_size,
                                epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbosity)
        if self.do_plots:
            self._training_history_plot(
                history_df=history,
                attribute='accuracy',
                filename="./training_history.eps")
            self._training_history_plot(
                history_df=history,
                attribute='loss',
                filename="./loss_history.eps")
        return (history.history['acc'][-1], history.history['val_acc'][-1])

    def predict(self, data):
        if self._clf == None:
            raise RuntimeError("Model has not been trained")
        prediction = self._clf.predict(
            data, batch_size=self.batch_size, verbose=self.verbosity)
        if self.output_dim == 1:
            return prediction.ravel()
        return np.argmax(prediction, axis=1)

    def evaluate(self, data, labels):
        if self._clf == None:
            raise RuntimeError("Model has not been trained")
        scores = self._clf.evaluate(
            data, labels, batch_size=self.batch_size, verbose=self.verbosity)
        idx = self._clf.metrics_names.index("acc")
        return scores[idx]

    def save_model(self, save_path):
        if self._clf == None:
            raise RuntimeError("Model has not been trained")
        model_json = self._clf.to_json()
        with open(save_path+'.json', "w") as json_file:
            json_file.write(model_json)
        self._clf.save_weights(save_path+'.h5')

    def load_model(self, load_path):
        json_file = open(load_path+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(load_path+'.h5')
        self._clf = loaded_model
        self._compile_neural_network()

    def _build_neural_network(self, n_samples):
        n_neurons = self._network_structure_heuristic(n_samples)
        neurons_per_layer = n_neurons / self.n_hidden_layers
        self._clf = Sequential()
        self._clf.add(Dense(neurons_per_layer, input_dim=self.input_dim))
        for i in range(self.n_hidden_layers-1):
            self._clf.add(Dense(neurons_per_layer, activation=tf.nn.relu))
        if self.output_dim == 1:
            self._clf.add(Dense(self.output_dim, activation=tf.nn.relu))
        else:
            self._clf.add(Dense(self.output_dim, activation=tf.nn.softmax))
        self._compile_neural_network()
        if self.verbosity >= 1:
            print(self._clf.summary())

    def _compile_neural_network(self):
        self._clf.compile(optimizer=self.optimizer,
                          loss=self.loss_function,
                          metrics=self.metrics)

    def _network_structure_heuristic(self, n_samples):
        ''' Based on this https://stats.stackexchange.com/a/136542 '''
        return n_samples / (self.alpha * (self.input_dim + self.output_dim))

    @staticmethod
    def _training_history_plot(history_df, attribute, filename):
        if attribute == 'accuracy':
            plt.plot(history_df.history['acc'])
            plt.plot(history_df.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
        elif attribute == 'loss':
            plt.plot(history_df.history['loss'])
            plt.plot(history_df.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(filename, format='eps')
        plt.close()
