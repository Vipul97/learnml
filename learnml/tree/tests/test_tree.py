from learnml.tree import DecisionTreeClassifier
from learnml.visuals.tree import draw_tree
import pandas as pd
import unittest


class TestDecisionTreeClassifier(unittest.TestCase):
    def test_fit(self):
        data = pd.read_csv('learnml/tree/tests/test_data.csv')

        tree_clf = DecisionTreeClassifier(mode='C45')
        tree = tree_clf.fit(list(data.keys()), data)
        # draw_tree(tree)


if __name__ == '__main__':
    unittest.main()
