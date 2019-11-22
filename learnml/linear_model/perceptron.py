import matplotlib.pyplot as plt
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Perceptron:
    def __init__(self, dataset, n_iters, lr, input=None):
        self.X = np.array([dataset[i][0] for i in range(len(dataset))])
        self.T = np.array([dataset[i][1] for i in range(len(dataset))])
        self.n_iters = n_iters
        self.lr = lr
        self.input = input
        self.size = len(dataset)
        self.w = np.random.rand(3)
        self.show_values(self.size, self.X, self.w, self.T)
        self.draw(self.size, self.X, self.w, self.T)
        self.train(self.n_iters, self.size, self.X, self.w, self.T, self.lr)
        self.draw_input(self.input, self.w)
        self.compute_error(self.size, self.X, self.w, self.T)
        plt.show()

    def show_values(self, size, X, w, T):
        print "Dataset:"
        for i in range(size):
            print "x =", X[i], "t =", T[i]
        print
        print "Weights:"
        print w

    def activation_function(self, a):
        return 1 if a >= 0 else -1

    def train(self, n_iters, size, X, w, T, lr):
        updates = 0

        print
        print "Training..."
        for n in range(n_iters):
            for i in range(size):
                if self.activation_function(np.dot(X[i], w)) * T[i] <= 0:
                    w += lr * X[i] * T[i]
                    self.draw(size, X, w, T)
                    updates += 1
        print "Training Complete!"
        print "Updates =", updates

    def test(self, x, w):
        return self.activation_function(np.dot(x, w))

    def compute_error(self, size, X, w, T):
        errors = 0

        for i in range(size):
            if self.activation_function(np.dot(X[i], w)) * T[i] <= 0:
                errors += 1
        print "Error: " + str(float(errors) / size * 100) + "%"

    def draw(self, size, X, w, T):
        plt.clf()
        plt.title('Perceptron')
        plt.xlabel('Red')
        plt.ylabel('Blue')
        plt.axis([-1.25, 1.25, -1.25, 1.25])

        for i in range(size):
            plt.scatter(X[i][1], X[i][2], c='red' if T[i] == 1 else 'blue',
                        alpha=1 if self.activation_function(np.dot(X[i], w)) * T[i] > 0 else 0.25)

        plt.plot([-1, 1], -(w[1] / w[2]) * np.array([-1, 1]) - (w[0] / w[2]), c='black')
        plt.pause(0.001)

    def draw_input(self, input, w):
        if input:
            for x in input:
                plt.scatter(x[1], x[2], c='red' if self.test(x, w) == 1 else 'blue', marker='*')


dataset = []
size = 200

for i in range(size):
    p = Point(np.random.random() * 2 - 1, np.random.random() * 2 - 1)
    dataset.append(np.array([np.array([1, p.x, p.y]), 1 if p.x >= p.y else -1]))

p = Perceptron(dataset, 1, 0.1, [[1, np.random.rand() * 2 - 1, np.random.rand() * 2 - 1] for i in range(5)])
