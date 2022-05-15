from learnml.tree import DecisionTreeClassifier
import numpy.testing
import pandas as pd
import unittest


class TestDecisionTreeClassifier(unittest.TestCase):
    def test_fit_predict(self):
        data = pd.read_csv('learnml/tree/tests/test_data.csv', dtype=object)

        X_train = data.drop('Vegetation', axis=1)
        y_train = data['Vegetation']

        for mode in ['ID3', 'C4.5']:
            with self.subTest(mode=mode):
                tree_clf = DecisionTreeClassifier(mode=mode)
                tree_clf.fit(X_train, y_train)
                y_pred = tree_clf.predict(X_train)

                numpy.testing.assert_equal(['carrot', 'radish', 'radish', 'carrot', 'coriander', 'coriander', 'carrot'],
                                           y_pred)


if __name__ == '__main__':
    unittest.main()
