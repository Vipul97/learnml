from learnml.utils import sigmoid
import numpy as np


class SGDClassifier:
    def __init__(self, penalty='l2', alpha=0.0001, max_iter=1000, eta0=0.0):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.eta0 = eta0
        self.coef_ = None
        self.costs = []
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X = np.c_[np.ones((n_samples, 1)), X]
        w = np.zeros(n_features + 1)

        for _ in range(self.max_iter):
            p = sigmoid(np.dot(X, w))
            cost = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            w -= self.eta0 * np.dot(X.T, p - y) / n_samples

            if self.penalty == 'l2':
                cost += (self.alpha / (2 * n_samples)) * np.dot(w[1:], w[1:])
                w[1:] -= self.alpha / n_samples * w[1:]

            elif self.penalty == 'l1':
                cost += self.alpha / n_samples * np.sum(np.abs(w[1:]))
                w[1:] -= 2 * self.alpha / n_samples * np.sign(w[1:])

            self.costs.append(cost)

        self.intercept_, *self.coef_ = w

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        p = sigmoid(np.dot(X, np.r_[self.intercept_, *self.coef_]))
        return (p > 0.5).astype(int)

    def predict_proba(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        p = np.array(sigmoid(np.dot(X, np.r_[self.intercept_, *self.coef_])))
        return np.column_stack((1 - p, p))


class SGDRegressor:
    def __init__(self, penalty='l2', alpha=0.0001, max_iter=1000, eta0=0.01):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.eta0 = eta0
        self.coef_ = None
        self.costs = []
        self.intercept_ = None
        self.w_path = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X = np.c_[np.ones((n_samples, 1)), X]
        w = np.random.randn(n_features + 1)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                prediction = np.dot(X[i], w)
                error = prediction - y[i]
                cost = error ** 2

                w -= self.eta0 * 2 * error * X[i]

                if self.penalty == 'l2':
                    cost += self.alpha * np.dot(w[1:], w[1:])
                    w[1:] -= 2 * self.alpha * w[1:]

                elif self.penalty == 'l1':
                    cost += self.alpha * np.sum(np.abs(w[1:]))
                    w[1:] -= 2 * self.alpha * np.sign(w[1:])

                self.costs.append(cost)
                self.w_path.append(w.copy())

        self.intercept_, *self.coef_ = w
        self.w_path = np.array(self.w_path)

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return np.dot(X, np.r_[self.intercept_, *self.coef_])
