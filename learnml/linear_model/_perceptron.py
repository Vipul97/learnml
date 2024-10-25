import numpy as np


class Perceptron:
    def __init__(self, max_iter=1000, eta0=1.0):
        self.max_iter = max_iter
        self.eta0 = eta0
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X = np.c_[np.ones((n_samples, 1)), X]
        w = np.zeros(n_features + 1)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                y_hat = np.heaviside(np.dot(X[i], w), 0)
                w += self.eta0 * (y[i] - y_hat) * X[i]

        self.intercept_, *self.coef_ = w

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return np.heaviside(np.dot(X, np.r_[self.intercept_, *self.coef_]), 0)
