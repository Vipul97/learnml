import numpy as np


class Perceptron:
    def __init__(self, max_iter=1000, eta0=1.0):
        self.__max_iter = max_iter
        self.__eta0 = eta0
        self.coef_ = None
        self.intercept_ = None
        self.n_iter = None
        self.t_ = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        X = np.c_[np.ones((n_samples, 1)), X]
        n_features = X.shape[1]
        self.intercept_, *self.coef_ = np.zeros(n_features)
        self.n_iter = self.__max_iter
        self.t_ = self.n_iter * n_samples
        w = np.array([self.intercept_, *self.coef_])

        for _ in range(self.n_iter):
            for i in range(n_samples):
                y_hat = np.heaviside(np.dot(X[i], w), 0)
                w += self.__eta0 * (y[i] - y_hat) * X[i]

        self.intercept_, *self.coef_ = w

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        w = np.array([self.intercept_, *self.coef_])

        return np.heaviside(np.dot(X, w), 0)
