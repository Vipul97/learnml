from learnml.linear_model import SGDClassifier, SGDRegressor
from learnml.model_selection import train_test_split
from learnml.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import unittest


class TestSGDClassifier(unittest.TestCase):
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

        for penalty in ['l2', 'l1']:
            with self.subTest(penalty=penalty):
                log_clf = SGDClassifier(penalty=penalty, max_iter=5000, eta0=1)
                log_clf.fit(X_train, y_train)
                y_pred = log_clf.predict(X_test)

                accuracy_score = np.mean(y_pred == y_test)
                np.testing.assert_(accuracy_score >= self.ACCURACY_THRESHOLD)


class TestSGDRegressor(unittest.TestCase):
    def test_fit_predict(self):
        X = np.array([[1], [2]])
        y = np.array([1, 2])

        for penalty in ['l2', 'l1']:
            with self.subTest(penalty=penalty):
                sgd_reg = SGDRegressor(penalty=penalty)
                sgd_reg.fit(X, y)

                np.testing.assert_almost_equal(sgd_reg.coef_, np.array([1]), 1)
                np.testing.assert_almost_equal(sgd_reg.intercept_, np.array([0]), 1)
                np.testing.assert_almost_equal(sgd_reg.predict(X), np.array([1, 2]), 1)


if __name__ == '__main__':
    unittest.main()
