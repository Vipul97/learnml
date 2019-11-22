import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    def __init__(self, filename, alpha, num_iters):
        print "Loading Data..."
        self.data = np.loadtxt(filename, delimiter=",")
        self.X = self.data[:, :-1]
        self.y = self.data[:, -1]
        self.show_data(self.X, self.y)
        print "Normalizing Features..."
        self.X, self.mu, self.rang = self.feature_normalize(self.X)
        self.X = np.insert(self.X, 0, 1, axis=1)
        print "Running gradient descent..."
        self.w = np.zeros(np.shape(self.X)[1])
        self.w, self.J_history = self.gradient_descent(self.X, self.y, self.w, alpha, num_iters)
        plt.title('Gradient Descent')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost J')
        plt.plot(np.arange(num_iters), self.J_history)
        print "Weights computed from gradient descent:"
        print self.w
        print
        print "Solving with normal equations..."
        self.w = self.normal_eqn(self.X, self.y)
        print "Weights computed from the normal equations:"
        print self.w
        plt.show()

    def show_data(self, X, y):
        print "Dataset:"
        for i in range(len(y)):
            print "x =", X[i], ", y =", y[i]
        print

    def feature_normalize(self, X):
        mu = np.mean(X, axis=0)
        rang = np.ptp(X, axis=0)

        return (X - mu) / rang, mu, rang

    def compute_cost(self, X, y, w):
        return np.sum((np.dot(X, w) - y) ** 2) / (2 * len(y))

    def gradient_descent(self, X, y, w, alpha, num_iters):
        J_history = np.zeros(num_iters)

        for iter in range(num_iters):
            w -= (alpha / len(y)) * np.dot(np.dot(X, w) - y, X)
            J_history[iter] = self.compute_cost(X, y, w)

        return w, J_history

    def normal_eqn(self, X, y):
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    def predict(self, input):
        y = np.dot(np.insert(self.feature_normalize(input)[0], 0, 1, axis=1), self.w)

        print
        print "Predictions:"
        for i in range(len(y)):
            print "x =", input[i], "y =", y[i]


lin_reg = LinearRegression('data.txt', 0.01, 400)
lin_reg.predict(np.array([[1650.0, 3.0], [1852.0, 4.0]]))
