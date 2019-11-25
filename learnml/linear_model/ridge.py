import numpy as np


class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.X = np.c_[np.ones((X.shape[0], 1)), X]
        self.y = y

        A = np.eye(self.X.shape[1])
        A[0][0] = 0

        self.intercept_, *self.coef_ = np.dot(np.linalg.inv(self.X.T @ self.X + self.alpha * A) @ self.X.T, self.y)

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]

        return np.dot(X, [self.intercept_, *self.coef_])
