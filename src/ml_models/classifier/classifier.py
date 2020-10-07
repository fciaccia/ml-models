from enum import Enum

from ml_models.classifier.classifier_svm import ClassifierSVM
from ml_models.classifier.classifier_dnn import ClassifierDNN


class ClassifierTypes(Enum):
    svm = 'svm'
    dnn = 'dnn'


DEFAULT_CLASSIFIER = ClassifierTypes.svm
DEFAULT_TEST_SIZE = 0.3


class Classifier():

    def __init__(self,
                 input_dim,
                 output_dim,
                 model=DEFAULT_CLASSIFIER,
                 **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = model
        if self.model == ClassifierTypes.svm:
            self.processor = ClassifierSVM(
                input_dim=self.input_dim, output_dim=self.output_dim, **kwargs)
        if self.model == ClassifierTypes.dnn:
            self.processor = ClassifierDNN(
                input_dim=self.input_dim, output_dim=self.output_dim, **kwargs)

    def train(self, data, labels,
              test_size=DEFAULT_TEST_SIZE):
        accuracy_train, accuracy_test = self.processor.train(
            data, labels, test_size)
        return (accuracy_train, accuracy_test)

    def predict(self, data):
        return self.processor.predict(data)

    def evaluate(self, data, labels):
        return self.processor.evaluate(data, labels)

    def save_model(self, save_path):
        self.processor.save_model(save_path)

    def load_model(self, load_path):
        self.processor.load_model(load_path)
