import numpy as np


class LinearRegression:
    def __init__(self):
        None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.w = np.linalg.pinv(np.c_[np.ones((self.X.shape[0], 1)), self.X]).dot(self.y)

    def predict(self, X):
        y_pred = np.dot(np.c_[np.ones((X.shape[0], 1)), X], self.w)

        return y_pred
