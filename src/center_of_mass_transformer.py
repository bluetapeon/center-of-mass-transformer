import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .calculators import prepare_weights, calc_center_of_mass, transform_from_cartesian_to_spherical


class CenterOfMassTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)

        weights = prepare_weights(y) if y is not None else np.ones(X.shape[1])

        self.cm_ = calc_center_of_mass(X, weights)

        return self

    def transform(self, X):
        check_is_fitted(self, 'cm_')
        X = check_array(X, accept_sparse=True)

        X_ex = X - self.cm_

        for _ in range(X.shape[1] - self.n_components):
            X_ex = transform_from_cartesian_to_spherical(X_ex)[:, 1:]

        return X_ex
