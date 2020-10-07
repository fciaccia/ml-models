import numpy as np
import pandas as pd

from enum import Enum

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class RepresentationTypes(Enum):
    pca = 'pca'
    standard = 'standard'
    normal = 'normal'


DEFAULT_REPRESENTATION = RepresentationTypes.pca
DEFAULT_PCA_VARIANCE = 0.9
DEFAULT_NORM = 'max'


class Representation():
    '''
    Data representation factory; it manipulates and change representation and/or
    provides decomposition (e.g. through PCA) to another base
    '''

    def __init__(self,
                 model=DEFAULT_REPRESENTATION,
                 **kwargs):
        super().__init__()
        self.model = model
        if self.model == RepresentationTypes.pca:
            self.processor = RepresentationPCA(**kwargs)
        if self.model == RepresentationTypes.standard:
            self.processor = RepresentationStandard(**kwargs)
        if self.model == RepresentationTypes.normal:
            self.processor = RepresentationNormal(**kwargs)

    def fit(self, data):
        if (not isinstance(data, pd.DataFrame) and
            not isinstance(data, pd.Series) and
            not isinstance(data, np.ndarray) and
                not isinstance(data, list)):
            raise TypeError("Not array like data")
        self.processor.fit(data)

    def transform(self, data):
        if (not isinstance(data, pd.DataFrame) and
            not isinstance(data, pd.Series) and
            not isinstance(data, np.ndarray) and
                not isinstance(data, list)):
            raise TypeError("Not array like data")
        return self.processor.transform(data)


class RepresentationPCA():

    def __init__(self,
                 pca_variance=DEFAULT_PCA_VARIANCE):
        self.pca_variance = pca_variance
        self.pca = PCA(self.pca_variance)

    def fit(self, data):
        self.pca.fit(data)

    def transform(self, data):
        return self.pca.transform(data)


class RepresentationStandard():

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)


class RepresentationNormal():

    def __init__(self, norm=DEFAULT_NORM):
        self.norm = norm
        self.normalizer = Normalizer(norm=self.norm)

    def fit(self, data):
        self.normalizer.fit(data)

    def transform(self, data):
        return self.normalizer.transform(data)
