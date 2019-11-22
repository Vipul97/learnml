import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork:
    def __init__(self, filename, num_layers, num_nodes, alpha, epochs, add_bias=True):
        self.data = np.loadtxt(filename, delimiter=",")
        self.X = self.data[:, :-1]
        self.m = len(self.X)
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.num_classes = num_nodes[-1]
        self.W = np.zeros(self.num_layers, dtype='object')
        self.alpha = alpha
        self.epochs = epochs
        self.add_bias = add_bias

        if self.num_classes > 1:
            self.y = np.zeros((self.m, self.num_classes))

            for i in range(self.m):
                self.y[i, int(self.data[i, -1])] = 1

        else:
            self.y = np.array([self.data[:, -1]]).T

        for i in range(1, self.num_layers):
            if add_bias:
                self.W[i] = np.random.rand(self.num_nodes[i - 1] + 1, self.num_nodes[i])
            else:
                self.W[i] = np.random.rand(self.num_nodes[i - 1], self.num_nodes[i])

        self.W, self.J_history = self.train(self.X, self.num_layers, self.num_classes, self.W, self.y, self.alpha,
                                            self.epochs, self.add_bias)

        self.H = self.predict(self.X, self.num_layers, self.num_classes, self.W, self.add_bias)
        print "Training Set Accuracy: ", np.mean(
            [float(np.array_equal(self.H[i], self.y[i])) for i in range(self.m)]) * 100

        plt.title('Gradient Descent')
        plt.xlabel('Number of epochs')
        plt.ylabel('Cost J')
        plt.plot(np.arange(self.epochs), self.J_history)
        plt.show()

    def append_bias(self, x, add_bias):
        return np.insert(x, 0, 1) if add_bias else x

    def sigmoid(self, z):
        return 1 / (1 + np.e ** -z)

    def grad_sigmoid(self, z):
        s = self.sigmoid(z)

        return s * (1 - s)

    def cost(self, h, y, m):
        return sum(sum((h - y) ** 2)) / (2 * m)

    def feedforward(self, X, num_layers, W, add_bias):
        a = np.zeros(num_layers + 1, dtype=object)
        z = np.zeros(num_layers + 1, dtype=object)

        a[1] = self.append_bias(X, add_bias)
        z[2] = np.dot(a[1], W[1])
        for i in range(2, num_layers):
            a[i] = self.append_bias(self.sigmoid(z[i]), add_bias)
            z[i + 1] = np.dot(a[i], W[i])

        a[num_layers] = self.sigmoid(z[num_layers])

        return a, z, a[-1]

    def backprop(self, num_layers, W, a, z, h, y, alpha, add_bias):
        delta = np.zeros(num_layers + 1, dtype=object)

        delta[num_layers] = np.array([(h - y) * self.grad_sigmoid(z[num_layers])]).T

        for i in range(num_layers - 1, 1, -1):
            if add_bias:
                delta[i] = np.dot(W[i], delta[i + 1])[1:] * np.array([self.grad_sigmoid(z[i])]).T
            else:
                delta[i] = np.dot(W[i], delta[i + 1]) * np.array([self.grad_sigmoid(z[i])]).T

        for i in range(num_layers - 1, 0, -1):
            W[i] -= alpha * np.dot(delta[i + 1], np.array([a[i]])).T

        return W

    def train(self, X, num_layers, num_classes, W, y, alpha, epochs, add_bias):
        m = len(X)
        J_history = np.zeros(epochs)

        for e in range(epochs):
            H = np.zeros((m, num_classes))

            for i in range(m):
                a, z, h = self.feedforward(X[i], num_layers, W, add_bias)

                W = self.backprop(num_layers, W, a, z, h, y[i], alpha, add_bias)
                H[i] = h

            J_history[e] = self.cost(H, y, m)
            print 'Epoch:', e, 'Cost:', J_history[e]

        return W, J_history

    def predict(self, X, num_layers, num_classes, W, add_bias):
        m = len(X)
        H = np.zeros((m, num_classes))

        for i in range(m):
            if num_classes > 1:
                H[i, np.argmax(self.feedforward(X[i], num_layers, W, add_bias)[-1])] = 1
            else:
                H[i] = self.feedforward(X[i], num_layers, W, add_bias)[-1]

        return H


nn = NeuralNetwork('data.txt', 2, [2, 2], 0.1, 10000, add_bias=True)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]

print
print "Predictions:"
for i in range(len(X)):
    print "x =", X[i], "y =", nn.predict([X[i]], nn.num_layers, nn.num_classes, nn.W, nn.add_bias)[0]
