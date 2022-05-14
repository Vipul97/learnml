from learnml.tree import DecisionTreeClassifier
import pandas as pd
import unittest


class TestDecisionTreeClassifier(unittest.TestCase):
    def test_fit(self):
        data = pd.read_csv('learnml/tree/tests/test_data.csv', dtype=object)

        X_train = data.drop('Vegetation', axis=1)
        y_train = data['Vegetation']

        tree_clf = DecisionTreeClassifier(mode='ID3')
        tree_clf.fit(X_train, y_train)


if __name__ == '__main__':
    unittest.main()
