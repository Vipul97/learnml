import numpy as np


class SGDRegressor:
    def __init__(self, penalty='l2', alpha=0.0001, max_iter=1000, eta0=0.01):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.eta0 = eta0

    def fit(self, X, y):
        def gradient_descent():
            def sign(w):
                w[w < 0] = -1
                w[w > 0] = 1

                return w

            costs, w_path = [], []
            w = [self.intercept_, *self.coef_]

            for iter in range(self.max_iter):
                for i in range(self.m):
                    cost = (np.dot(self.X[i], w) - y[i]) ** 2
                    w -= self.eta0 * 2 * np.dot(self.X[i].T, np.dot(self.X[i], w) - y[i])

                    if self.penalty == 'l2':
                        cost += self.alpha * np.dot(w[1:].T, w[1:])
                        w[1:] -= 2 * self.alpha * w[1:]
                    elif self.penalty == 'l1':
                        cost += self.alpha * np.sum(np.abs(w[1:]))
                        w[1:] -= 2 * self.alpha * sign(w[1:])

                    costs.append(cost)
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
