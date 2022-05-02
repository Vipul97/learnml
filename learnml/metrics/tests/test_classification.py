from learnml.metrics import confusion_matrix
import numpy as np
import numpy.testing
import unittest


class Test(unittest.TestCase):
    def test_confusion_matrix(self):
        expected_results = [
            np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]),
            np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])]

        for i, y_pred in enumerate(np.array([[1, 2, 3, 4, 5], [1, 1, 2, 2, 3]])):
            numpy.testing.assert_equal(expected_results[i], confusion_matrix(np.array([1, 2, 3, 4, 5]), y_pred))


if __name__ == '__main__':
    unittest.main()
