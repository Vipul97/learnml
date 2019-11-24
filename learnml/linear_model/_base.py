import numpy as np


class LinearRegression:
    def __init__(self):
        None

    def fit(self, X, y):
        self.X = np.c_[np.ones((X.shape[0], 1)), X]
        self.y = y
        self.intercept_, *self.coef_ = np.linalg.pinv(self.X).dot(self.y)

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]

        return np.dot(X, [self.intercept_, *self.coef_])
