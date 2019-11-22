import numpy as np


class StandardScaler:
    def __init__(self):
        self.u = 0
        self.s = 0

    def fit(self, X):
        self.u = np.mean(X, axis=0)
        self.s = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.u) / self.s
