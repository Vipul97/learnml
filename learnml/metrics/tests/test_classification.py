from learnml.metrics import confusion_matrix
import numpy as np
import unittest


class TestConfusionMatrix(unittest.TestCase):
    def setUp(self):
        self.test_cases = [
            {
                'y_true': np.array([1, 2, 3, 4, 5]),
                'y_pred': np.array([1, 2, 3, 4, 5]),
                'expected': np.array([[1, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 1]])
            },
            {
                'y_true': np.array([1, 2, 3, 4, 5]),
                'y_pred': np.array([1, 1, 2, 2, 3]),
                'expected': np.array([[1, 0, 0],
                                      [1, 0, 0],
                                      [0, 1, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])
            }
        ]

    def test_confusion_matrix(self):
        for case in self.test_cases:
            with self.subTest(y_pred=case['y_pred']):
                result = confusion_matrix(case['y_true'], case['y_pred'])
                np.testing.assert_array_equal(result, case['expected'])


if __name__ == '__main__':
    unittest.main()
