import numpy as np


class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.n_iter_ = None

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]

        A = np.eye(X.shape[1])
        A[0][0] = 0

        self.intercept_, *self.coef_ = np.dot(np.linalg.inv(X.T @ X + self.alpha * A) @ X.T, y)

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        w = [self.intercept_, *self.coef_]

        return np.dot(X, w)
