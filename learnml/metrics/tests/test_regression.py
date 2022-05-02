from learnml.metrics import mean_squared_error
import numpy as np
import unittest


class Test(unittest.TestCase):
    def test_mean_squared_error(self):
        expected_results = [0, 1]

        for i, y_pred in enumerate(np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])):
            self.assertEqual(expected_results[i], mean_squared_error(np.array([1, 2, 3, 4, 5]), y_pred))


if __name__ == '__main__':
    unittest.main()
