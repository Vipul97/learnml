from learnml.linear_model import SGDClassifier, SGDRegressor
from learnml.model_selection import train_test_split
from learnml.preprocessing import StandardScaler
import numpy as np
import numpy.testing
import pandas as pd
import unittest


class TestSGDClassifier(unittest.TestCase):
    def test_fit_predict(self):
        data = pd.read_csv('learnml/linear_model/tests/test_data.csv')

        train, test = train_test_split(data, test_size=0.2, random_state=42)

        X_train = train.drop('y', axis=1).values
        y_train = train['y'].values
        X_test = test.drop('y', axis=1).values
        y_test = test['y'].values

        scaler = StandardScaler()

        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        for penalty in ['l2', 'l1']:
            with self.subTest(penalty=penalty):
                log_clf = SGDClassifier(penalty=penalty, max_iter=5000, eta0=1)
                log_clf.fit(X_train, y_train)
                y_pred = log_clf.predict(X_test)

                accuracy_score = np.mean(y_pred == y_test)
                self.assertTrue(accuracy_score >= 0.8)


class TestSGDRegressor(unittest.TestCase):
    def test_fit_predict(self):
        X = np.array([[1], [2]])
        y = np.array([1, 2])

        for penalty in ['l2', 'l1']:
            with self.subTest(penalty=penalty):
                sgd_reg = SGDRegressor(penalty=penalty)
                sgd_reg.fit(X, y)

                numpy.testing.assert_almost_equal(np.array([1]), sgd_reg.coef_, 1)
                numpy.testing.assert_almost_equal(np.array([0]), sgd_reg.intercept_, 1)
                numpy.testing.assert_almost_equal(np.array([1, 2]), sgd_reg.predict(X), 1)


if __name__ == '__main__':
    unittest.main()
