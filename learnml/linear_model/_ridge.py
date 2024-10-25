import numpy as np


class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]

        A = np.eye(X.shape[1])
        A[0][0] = 0

        w = np.linalg.inv(X.T @ X + self.alpha * A) @ X.T @ y
        self.intercept_ = w[0]
        self.coef_ = w[1:]

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return X @ np.r_[self.intercept_, self.coef_]
