import numpy as np


class SGDRegressor:
    def __init__(self, max_iter=1000, eta0=0.01):
        self.max_iter = max_iter
        self.eta0 = eta0

    def fit(self, X, y):
        def gradient_descent():
            costs, w_path = [], []
            w = [self.intercept_, *self.coef_]

            for iter in range(self.max_iter):
                for i in range(self.m):
                    w -= self.eta0 * 2 * np.dot(self.X.T, np.dot(self.X, w) - y)
                    w_path.append(w.copy())

            self.intercept_, *self.coef_ = w

            return costs, np.array(w_path)

        self.X = np.c_[np.ones((X.shape[0], 1)), X]
        self.y = y
        self.m = self.X.shape[0]
        self.intercept_, *self.coef_ = np.random.randn(self.X.shape[1])
        self.costs, self.w_path = gradient_descent()

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]

        return np.dot(X, [self.intercept_, *self.coef_])
