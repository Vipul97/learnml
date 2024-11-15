from learnml.linear_model import Perceptron
from learnml.model_selection import train_test_split
from learnml.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import unittest


class TestPerceptron(unittest.TestCase):
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    ACCURACY_THRESHOLD = 0.8

    def setUp(self):
        self.data = pd.read_csv('learnml/linear_model/tests/test_data.csv')

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

        per_clf = Perceptron()
        per_clf.fit(X_train, y_train)
        y_pred = per_clf.predict(X_test)

        accuracy_score = np.mean(y_pred == y_test)
        np.testing.assert_(accuracy_score >= self.ACCURACY_THRESHOLD)


if __name__ == '__main__':
    unittest.main()
