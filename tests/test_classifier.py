import numpy as np
import pandas as pd
import pytest

from numpy.random import RandomState
from pandas.util.testing import assert_frame_equal
from sklearn.model_selection import train_test_split

from tech_nn_control.ml_models.classifier import Classifier
from tech_nn_control.ml_models.classifier import ClassifierTypes

TEST_GAMMA = 'scale'
TEST_MODEL_DIR = '/tmp/pytest_model'


@pytest.fixture
def test_data():
    prng = RandomState(42)
    return pd.DataFrame(prng.randint(0, 10, size=(10, 4)))


@pytest.fixture
def test_labels():
    prng = RandomState(42)
    return pd.DataFrame(prng.randint(0, 2, size=(10, 1)))


@pytest.fixture
def svm_train_result():
    # Score computed with sklearn SVM/SVC implementation over fixtures
    return (1.0, 0.9)


@pytest.fixture
def svm_evaluate_result():
    # Score computed with sklearn SVM/SVC implementation over fixtures
    return 0.9


def test_svm_train_evaluate(test_data, test_labels, svm_train_result, svm_evaluate_result):
    classifier = Classifier(input_dim=4, output_dim=2,
                            model=ClassifierTypes.svm)
    train_res = classifier.train(
        data=test_data, labels=test_labels, test_size=0)
    eval_res = classifier.evaluate(test_data, test_labels)
    assert svm_train_result == train_res
    assert svm_evaluate_result == eval_res


def test_svm_train_multidimensional_labels(test_data):
    classifier = Classifier(input_dim=4, output_dim=2,
                            model=ClassifierTypes.svm)
    test_labels = pd.DataFrame(np.random.randint(0, 2, size=(10, 3)))
    with pytest.raises(ValueError):
        classifier.train(data=test_data, labels=test_labels, test_size=0)


def test_svm_train_not_unique_labels(test_data):
    classifier = Classifier(input_dim=4, output_dim=2,
                            model=ClassifierTypes.svm)
    test_labels = pd.DataFrame(np.random.randint(0, 1, size=(10, 1)))
    with pytest.raises(ValueError):
        classifier.train(data=test_data, labels=test_labels, test_size=0)


def test_svm_save_load_model(test_data, test_labels, svm_evaluate_result):
    classifier = Classifier(input_dim=4, output_dim=2,
                            model=ClassifierTypes.svm)
    classifier.train(data=test_data, labels=test_labels, test_size=0)
    classifier.save_model(TEST_MODEL_DIR)
    classifier.processor._clf = None
    classifier.load_model(TEST_MODEL_DIR)
    assert svm_evaluate_result == classifier.evaluate(test_data, test_labels)
