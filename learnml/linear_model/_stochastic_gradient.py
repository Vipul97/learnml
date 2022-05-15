from learnml.utils import sigmoid
import numpy as np


class SGDClassifier:
    def __init__(self, penalty='l2', alpha=0.0001, max_iter=1000, eta0=0.0):
        self.__penalty = penalty
        self.__alpha = alpha
        self.__max_iter = max_iter
        self.__eta0 = eta0
        self.coef_ = None
        self.costs = None
        self.intercept_ = None
        self.n_iter_ = None
        self.t_ = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        X = np.c_[np.ones((n_samples, 1)), X]
        n_features = X.shape[1]
        self.intercept_, *self.coef_ = np.zeros(n_features)
        self.n_iter_ = self.__max_iter
        self.t_ = self.n_iter_ * n_samples
        self.costs = []
        w = np.array([self.intercept_, *self.coef_])

        for _ in range(self.n_iter_):
            p = sigmoid(np.dot(X, w))
            cost = -1 / n_samples * (np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
            w -= self.__eta0 * 1 / n_samples * np.dot(X.T, p - y)

            if self.__penalty == 'l2':
                cost += (self.__alpha / (2 * n_samples)) * np.dot(w[1:].T, w[1:])
                w[1:] -= self.__alpha / n_samples * w[1:]

            elif self.__penalty == 'l1':
                cost += self.__alpha / n_samples * np.sum(np.abs(w[1:]))
                w[1:] -= 2 * self.__alpha / n_samples * np.sign(w[1:])

            self.costs.append(cost)

        self.intercept_, *self.coef_ = w

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred = np.zeros(X.shape[0])
        w = np.array([self.intercept_, *self.coef_])
        p = sigmoid(np.dot(X, w))
        y_pred[p > 0.5] = 1

        return y_pred

    def predict_proba(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        w = np.array([self.intercept_, *self.coef_])
        p = np.array(sigmoid(np.dot(X, w))).reshape(-1, 1)

        return np.concatenate([1 - p, p], axis=1)


class SGDRegressor:
    def __init__(self, penalty='l2', alpha=0.0001, max_iter=1000, eta0=0.01):
        self.__penalty = penalty
        self.__alpha = alpha
        self.__max_iter = max_iter
        self.__eta0 = eta0
        self.coef_ = None
        self.costs = None
        self.intercept_ = None
        self.n_iter_ = None
        self.t_ = None
        self.w_path = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        X = np.c_[np.ones((n_samples, 1)), X]
        n_features = X.shape[1]
        self.intercept_, *self.coef_ = np.random.randn(n_features)
        self.n_iter_ = self.__max_iter
        self.t_ = self.n_iter_ * n_samples
        self.costs, self.w_path = [], []
        w = np.array([self.intercept_, *self.coef_])

        for _ in range(self.n_iter_):
            for i in range(n_samples):
                cost = (np.dot(X[i], w) - y[i]) ** 2
                w -= self.__eta0 * 2 * np.dot(X[i].T, np.dot(X[i], w) - y[i])

                if self.__penalty == 'l2':
                    cost += self.__alpha * np.dot(w[1:].T, w[1:])
                    w[1:] -= 2 * self.__alpha * w[1:]

                elif self.__penalty == 'l1':
                    cost += self.__alpha * np.sum(np.abs(w[1:]))
                    w[1:] -= 2 * self.__alpha * np.sign(w[1:])

                self.costs.append(cost)
                self.w_path.append(w.copy())

        self.intercept_, *self.coef_ = w
        self.w_path = np.array(self.w_path)

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        w = np.array([self.intercept_, *self.coef_])

        return np.dot(X, w)
