import numpy as np


class NeuralNetwork:
    def __init__(self, layer_dims=np.array([1]), learning_rate=1.0, num_iterations=1000):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.costs = []
        self.m = None
        self.parameters = {}
        self.X = None
        self.y = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z)), z

    def _relu(self, z):
        return np.maximum(0, z), z

    def _sigmoid_derivative(self, da, cache):
        s = 1 / (1 + np.exp(-cache))
        return da * s * (1 - s)

    def _relu_derivative(self, da, cache):
        dz = np.array(da, copy=True)
        dz[cache <= 0] = 0
        return dz

    def _initialize_parameters(self):
        parameters = {}
        L = len(self.layer_dims)
        for l in range(1, L):
            parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.01
            parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
        return parameters

    def _linear_forward(self, a, W, b):
        return np.dot(W, a) + b, (a, W, b)

    def _linear_activation_forward(self, a_prev, W, b, activation):
        z, linear_cache = self._linear_forward(a_prev, W, b)
        if activation == 'sigmoid':
            a, activation_cache = self._sigmoid(z)
            return a, (linear_cache, activation_cache)
        elif activation == 'relu':
            a, activation_cache = self._relu(z)
            return a, (linear_cache, activation_cache)

    def _compute_cost(self, aL):
        return -np.mean(self.y * np.log(aL) + (1 - self.y) * np.log(1 - aL))

    def _linear_backward(self, dz, cache):
        a_prev, W, _ = cache
        dW = np.dot(dz, a_prev.T) / self.m
        db = np.sum(dz, axis=1, keepdims=True) / self.m
        da_prev = np.dot(W.T, dz)
        return da_prev, dW, db

    def _linear_activation_backward(self, da, cache, activation):
        linear_cache, activation_cache = cache
        if activation == 'relu':
            return self._linear_backward(self._relu_derivative(da, activation_cache), linear_cache)
        elif activation == 'sigmoid':
            return self._linear_backward(self._sigmoid_derivative(da, activation_cache), linear_cache)

    def feedforward(self, X):
        caches = []
        a = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            a_prev = a
            a, cache = self._linear_activation_forward(a_prev, self.parameters[f'W{l}'], self.parameters[f'b{l}'],
                                                       'relu')
            caches.append(cache)

        aL, cache = self._linear_activation_forward(a, self.parameters[f'W{L}'], self.parameters[f'b{L}'], 'sigmoid')
        caches.append(cache)

        return aL, caches

    def _backprop(self, aL, caches):
        grads = {}
        L = len(caches)
        daL = - (np.divide(self.y, aL) - np.divide(1 - self.y, 1 - aL))

        current_cache = caches[L - 1]
        grads[f'da{L - 1}'], grads[f'dW{L}'], grads[f'db{L}'] = self._linear_activation_backward(daL, current_cache,
                                                                                                 'sigmoid')

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            grads[f'da{l}'], grads[f'dW{l + 1}'], grads[f'db{l + 1}'] = self._linear_activation_backward(
                grads[f'da{l + 1}'], current_cache, 'relu')

        return grads

    def _update_parameters(self, grads):
        L = len(self.parameters) // 2
        for l in range(L):
            self.parameters[f'W{l + 1}'] -= self.learning_rate * grads[f'dW{l + 1}']
            self.parameters[f'b{l + 1}'] -= self.learning_rate * grads[f'db{l + 1}']

    def fit(self, X, y):
        self.X = X.T
        self.m = self.X.shape[1]
        self.y = y.reshape(1, self.m)
        self.parameters = self._initialize_parameters()

        for _ in range(self.num_iterations):
            aL, caches = self.feedforward(self.X)
            cost = self._compute_cost(aL)
            grads = self._backprop(aL, caches)
            self._update_parameters(grads)
            self.costs.append(cost)

    def predict(self, X):
        aL, _ = self.feedforward(X.T)
        return aL > 0.5
