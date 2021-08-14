from learnml.linear_model import Perceptron
from learnml.model_selection import train_test_split
from learnml.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import unittest


class TestPerceptron(unittest.TestCase):
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

        per_clf = Perceptron()
        per_clf.fit(X_train, y_train)
        y_pred = per_clf.predict(X_test)

        accuracy_score = np.mean(y_pred == y_test)
        self.assertTrue(accuracy_score >= 0.8)


if __name__ == '__main__':
    unittest.main()
