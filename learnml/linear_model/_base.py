import numpy as np


class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        weights = np.linalg.pinv(X).dot(y)
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]

        return np.dot(X, np.r_[self.intercept_, *self.coef_])
