import numpy as np


class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.intercept_, *self.coef_ = np.linalg.pinv(X).dot(y)

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        w = np.array([self.intercept_, *self.coef_])

        return np.dot(X, w)
