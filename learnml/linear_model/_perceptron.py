import numpy as np


class Perceptron:
    def __init__(self, max_iter=1000, eta0=1.0):
        self.max_iter = max_iter
        self.eta0 = eta0

    def heaviside(self, z):
        return 1 if z >= 0 else 0

    def fit(self, X, y):
        def initialize_parameters(dim):
            w = np.zeros((dim, 1))
            b = 0

            return {"w": w, "b": b}

        def optimize():
            for i in range(self.m):
                x_i = self.X[:, i]
                y_i = self.y[:, i]
                y_hat = self.heaviside(np.dot(x_i, self.parameters["w"]) + self.parameters["b"])

                self.parameters["w"] += self.eta0 * (y_i - y_hat) * x_i.reshape((self.X.shape[0], 1))
                self.parameters["b"] += self.eta0 * (y_i - y_hat)

        self.X = X.T
        self.m = self.X.shape[1]
        self.y = y.reshape(1, self.m)
        self.parameters = initialize_parameters(self.X.shape[0])

        for i in range(self.max_iter):
            optimize()

    def predict(self, X):
        X = X.T
        y_pred = np.zeros((1, X.shape[1]))

        for i in range(X.shape[1]):
            y_pred[0, i] = self.heaviside(np.dot(self.X[:, i], self.parameters["w"]) + self.parameters["b"])

        return y_pred
