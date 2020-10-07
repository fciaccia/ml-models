import numpy as np
import pandas as pd
import pytest

from numpy.random import RandomState
from pandas.util.testing import assert_frame_equal

from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

from tech_nn_control.ml_models.representation import Representation
from tech_nn_control.ml_models.representation import RepresentationTypes

TEST_PCA_VARIANCE = 0.7
TEST_NORM = 'l2'


@pytest.fixture
def test_data():
    prng = RandomState(42)
    return pd.DataFrame(prng.randint(0, 10, size=(10, 4)))


@pytest.fixture
def pca_result(test_data):
    pca = PCA(TEST_PCA_VARIANCE)
    return pd.DataFrame(pca.fit_transform(test_data))


@pytest.fixture
def standard_scaler_result(test_data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(test_data))


@pytest.fixture
def normalizer_result(test_data):
    normalizer = Normalizer(norm=TEST_NORM)
    return pd.DataFrame(normalizer.fit_transform(test_data))


def test_pca_fit_transform(test_data, pca_result):
    representation = Representation(
        model=RepresentationTypes.pca,
        pca_variance=TEST_PCA_VARIANCE)
    representation.fit(test_data)
    mod_data = representation.transform(test_data)
    assert_frame_equal(pd.DataFrame(mod_data), pca_result)


def test_standard_fit_transform(test_data, standard_scaler_result):
    representation = Representation(
        model=RepresentationTypes.standard)
    representation.fit(test_data)
    mod_data = representation.transform(test_data)
    assert_frame_equal(pd.DataFrame(mod_data), standard_scaler_result)


def test_normal_fit_transform(test_data, normalizer_result):
    representation = Representation(
        model=RepresentationTypes.normal,
        norm=TEST_NORM)
    representation.fit(test_data)
    mod_data = representation.transform(test_data)
    assert_frame_equal(pd.DataFrame(mod_data), normalizer_result)
