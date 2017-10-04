"""Code for maximum covariance analysis regression
"""
import numpy as np
from scipy.linalg import svd, lstsq
from sklearn.base import RegressorMixin, TransformerMixin, BaseEstimator


class MCA(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=4, demean=True):
        self.n_components = n_components
        self.demean = demean

    def fit(self, XY, y=None):

        # This is basically a hack to work around the difficult
        # with using scaled output for MCA but not for the linear regression
        assert len(XY) == 2

        X, Y = XY

        X = np.asarray(X)
        Y = np.asarray(Y)

        m, n = X.shape
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = Y.mean(axis=0)

        if self.demean:
            X = X - self.x_mean_
            Y = Y - self.y_mean_


        self.cov_ = X.T.dot(Y)/(m-1)  # covariance matrix
        U, S, Vt = svd(self.cov_, full_matrices=False)

        self._u = U
        self._v = Vt.T

        # x_scores = self.transform(X)
        # b = lstsq(x_scores, Y)[0]
        # self.coef_ = self.x_components_ @ b

        return self

    def transform(self, XY, y=None):
        assert len(XY) == 2
        X, Y = XY
        X = np.asarray(X)
        Y = np.asarray(Y)
        x_scores = (X-self.x_mean_).dot(self.x_components_)
        # if y is not None:
        #     y_scores = (Y-self.y_mean_).dot(self.y_components_)
        #     return x_scores, y_scores
        # else:
        return x_scores

    @property
    def x_components_(self):
        return self._u[:, :self.n_components]

    @property
    def y_components_(self):
        return self._v[:, :self.n_components]