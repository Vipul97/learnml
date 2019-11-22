import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    def __init__(self, filename, alpha, iterations, l):
        print "Plotting Data..."
        self.data = np.loadtxt(filename, delimiter=",")
        self.X = self.data[:, :-1]
        self.y = self.data[:, -1]
        self.plot_data(self.X, self.y)
        self.X = np.insert(self.data[:, :-1], 0, 1, axis=1)
        self.w = np.zeros(np.shape(self.X)[1])
        self.alpha = alpha
        self.iterations = iterations
        self.l = l
        print "Running Gradient Descent..."
        self.w, self.J_history = self.gradient_descent(self.X, self.y, self.w, self.alpha, self.iterations, self.l)
        print "Weights found by gradient descent:"
        print self.w
        self.x_plot = np.insert([[i] for i in np.arange(min(self.X[:, 1]), max(self.X[:, 1]), 0.01)], 0, 1, axis=1)
        plt.plot(self.x_plot[:, 1], np.dot(self.x_plot, self.w), c='red')
        plt.subplot(1, 2, 2)
        plt.title('Gradient Descent')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost J')
        plt.plot(np.arange(iterations), self.J_history)
        plt.show()

    def plot_data(self, X, y):
        plt.subplot(1, 2, 1)
        plt.title('Linear Regression')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.axis([min(X), max(X), min(y), max(y)])
        plt.scatter(X, y)

    def compute_cost(self, X, y, w):
        return np.sum((np.dot(X, w) - y) ** 2) / (2 * len(y))

    def gradient_descent(self, X, y, w, alpha, iterations, l):
        J_history = np.zeros(iterations)

        for iter in range(iterations):
            w -= (alpha / len(y)) * (np.dot(np.dot(X, w) - y, X) + (l * w))
            J_history[iter] = self.compute_cost(X, y, w)

        return w, J_history

    def predict(self, input):
        y = np.dot(input, self.w)

        print
        print "Predictions:"
        for i in range(len(y)):
            print "x =", input[i][1], "y =", y[i]


lin_reg = LinearRegression('data.txt', 0.01, 1500, 500)
lin_reg.predict(np.array([[1, 3.5], [1, 7]]))
