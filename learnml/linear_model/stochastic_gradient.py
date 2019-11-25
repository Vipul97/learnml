from learnml.utils import sigmoid, sign
import numpy as np


class SGDClassifier:
    def __init__(self, penalty='l2', alpha=0.0001, max_iter=1000, eta0=0.0):
        self.__penalty = penalty
        self.__alpha = alpha
        self.n_iter_ = max_iter
        self.__eta0 = eta0

    def fit(self, X, y):
        def gradient_descent():
            costs = []
            w = [self.intercept_, *self.coef_]

            for iter in range(self.n_iter_):
                p = sigmoid(np.dot(X, w))

                cost = -1 / m * (np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
                w -= self.__eta0 * 1 / m * np.dot(X.T, p - y)

                if self.__penalty == 'l2':
                    cost += (self.__alpha / (2 * m)) * np.dot(w[1:].T, w[1:])
                    w[1:] -= self.__alpha / m * w[1:]
                elif self.__penalty == 'l1':
                    cost += self.__alpha / m * np.sum(np.abs(w[1:]))
                    w[1:] -= 2 * self.__alpha / m * sign(w[1:])

                costs.append(cost)

            self.intercept_, *self.coef_ = w

            return costs

        X = np.c_[np.ones((X.shape[0], 1)), X]
        m = X.shape[0]
        self.intercept_, *self.coef_ = np.zeros(X.shape[1])
        self.costs = gradient_descent()

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred = np.zeros(X.shape[0])
        w = [self.intercept_, *self.coef_]

        p = sigmoid(np.dot(X, w))
        y_pred[p > 0.5] = 1

        return y_pred

    def predict_proba(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        w = [self.intercept_, *self.coef_]

        p = np.array(sigmoid(np.dot(X, w))).reshape(-1, 1)

        return np.concatenate([1 - p, p], axis=1)


class SGDRegressor:
    def __init__(self, penalty='l2', alpha=0.0001, max_iter=1000, eta0=0.01):
        self.__penalty = penalty
        self.__alpha = alpha
        self.n_iter_ = max_iter
        self.__eta0 = eta0

    def fit(self, X, y):
        def gradient_descent():
            costs, w_path = [], []
            w = [self.intercept_, *self.coef_]

            for iter in range(self.n_iter_):
                for i in range(m):
                    cost = (np.dot(X[i], w) - y[i]) ** 2
                    w -= self.__eta0 * 2 * np.dot(X[i].T, np.dot(X[i], w) - y[i])

                    if self.__penalty == 'l2':
                        cost += self.__alpha * np.dot(w[1:].T, w[1:])
                        w[1:] -= 2 * self.__alpha * w[1:]
                    elif self.__penalty == 'l1':
                        cost += self.__alpha * np.sum(np.abs(w[1:]))
                        w[1:] -= 2 * self.__alpha * sign(w[1:])

                    costs.append(cost)
                    w_path.append(w.copy())

            self.intercept_, *self.coef_ = w

            return costs, np.array(w_path)

        X = np.c_[np.ones((X.shape[0], 1)), X]
        m = X.shape[0]
        self.intercept_, *self.coef_ = np.random.randn(X.shape[1])
        self.costs, self.w_path = gradient_descent()

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]

        return np.dot(X, [self.intercept_, *self.coef_])
