import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


DEFAULT_SVC_GAMMA = 'scale'


class ClassifierSVM():

    def __init__(self, input_dim, output_dim,
                 gamma=DEFAULT_SVC_GAMMA,
                 do_prints=False,
                 **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.verbosity = do_prints
        self._clf = None

    def train(self, data, labels, test_size):
        if type(labels) == type(pd.DataFrame()):
            if len(labels.columns) > 1:
                raise ValueError("Multidimensional labels are not supported")
            else:
                labels = pd.Series(labels[labels.columns[0]])

        if len(set(labels.unique())) != self.output_dim:
            raise ValueError(
                "Indicated output_dim %d does not match with unique labels %d" % (self.output_dim, len(set(labels))))
        if test_size != 0:
            data_train, data_test, labels_train, labels_test = train_test_split(
                data, labels, test_size=test_size)
        else:
            data_train = data_test = data
            labels_train = labels_test = labels

        self._clf = SVC(gamma=self.gamma, verbose=self.verbosity)
        self._clf.fit(data_train, labels_train)
        # SVM classifier does not provide a training accuracy, returning 1
        return (1.0, self._clf.score(data_test, labels_test))

    def predict(self, data):
        if self._clf == None:
            raise RuntimeError("Model has not been trained")
        return self._clf.predict(data)

    def evaluate(self, data, labels):
        if self._clf == None:
            raise RuntimeError("Model has not been trained")
        return self._clf.score(data, labels)

    def save_model(self, save_path):
        if self._clf == None:
            raise RuntimeError("Model has not been trained")
        with open(save_path+'.svm', 'wb') as output_file:
            pickle.dump(self._clf, output_file)

    def load_model(self, load_path):
        with open(load_path+'.svm', 'rb') as input_file:
            self._clf = pickle.load(input_file)
