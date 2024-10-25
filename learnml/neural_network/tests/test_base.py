from learnml.model_selection import train_test_split
from learnml.neural_network import NeuralNetwork
from learnml.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import unittest


class TestNeuralNetwork(unittest.TestCase):
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    ACCURACY_THRESHOLD = 0.8
    LAYER_DIMS = np.array([2, 2, 1])
    LEARNING_RATE = 1
    NUM_ITERATIONS = 100

    def setUp(self):
        self.data = pd.read_csv('learnml/neural_network/tests/test_data.csv')

    def preprocess_data(self):
        train, test = train_test_split(self.data, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE)

        X_train = train.drop('y', axis=1).values
        y_train = train['y'].values
        X_test = test.drop('y', axis=1).values
        y_test = test['y'].values

        scaler = StandardScaler()

        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, y_train, X_test, y_test

    def test_fit_predict(self):
        X_train, y_train, X_test, y_test = self.preprocess_data()

        nn_clf = NeuralNetwork(layer_dims=self.LAYER_DIMS, learning_rate=self.LEARNING_RATE,
                               num_iterations=self.NUM_ITERATIONS)
        nn_clf.fit(X_train, y_train)
        y_pred = nn_clf.predict(X_test)

        accuracy_score = np.mean(y_pred == y_test)
        np.testing.assert_(accuracy_score >= self.ACCURACY_THRESHOLD)


if __name__ == '__main__':
    unittest.main()
