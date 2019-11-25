from learnml.utils import sigmoid
import numpy as np


class LogisticRegression:
    def __init__(self, max_iter=100, eta0=0.1):
        self.max_iter = max_iter
        self.eta0 = eta0

    def fit(self, X, y):
        def initialize_parameters(dim):
            w = np.zeros((dim, 1))
            b = 0

            return {"w": w, "b": b}

        def propagate():
            a = sigmoid(np.dot(self.parameters["w"].T, self.X) + self.parameters["b"])

            return {"dw": 1 / self.m * np.dot(self.X, (a - self.y).T),
                    "db": 1 / self.m * np.sum(a - self.y)}, \
                   -1 / self.m * (np.sum(self.y * np.log(a) + (1 - self.y) * np.log(1 - a)))

        def optimize():
            costs = []

            for iter in range(self.max_iter):
                grads, cost = propagate()

                self.parameters["w"] -= self.eta0 * grads["dw"]
                self.parameters["b"] -= self.eta0 * grads["db"]

                costs.append(cost)

            return costs

        self.X = X.T
        self.m = self.X.shape[1]
        self.y = y.reshape(1, self.m)
        self.parameters = initialize_parameters(self.X.shape[0])
        self.costs = optimize()

    def predict(self, X):
        X = X.T
        y_pred = np.zeros((1, X.shape[1]))
        a = sigmoid(np.dot(self.parameters["w"].T, X) + self.parameters["b"])

        for i in range(a.shape[1]):
            y_pred[0, i] = 0 if a[0, i] <= 0.5 else 1

        return y_pred
