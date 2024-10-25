from learnml.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import unittest


class TestDecisionTreeClassifier(unittest.TestCase):
    def setUpClass(cls):
        cls.data = pd.read_csv('learnml/tree/tests/test_data.csv', dtype=object)
        cls.X_train = cls.data.drop('Vegetation', axis=1)
        cls.y_train = cls.data['Vegetation']

    def test_fit_predict(self):
        expected_predictions = ['carrot', 'radish', 'radish', 'carrot', 'coriander', 'coriander', 'carrot']

        for mode in ['ID3', 'C4.5']:
            with self.subTest(mode=mode):
                tree_clf = DecisionTreeClassifier(mode=mode)
                tree_clf.fit(self.X_train, self.y_train)
                y_pred = tree_clf.predict(self.X_train)

                np.testing.assert_equal(y_pred, expected_predictions)


if __name__ == '__main__':
    unittest.main()
