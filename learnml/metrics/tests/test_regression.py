from learnml.metrics import mean_squared_error
import numpy as np
import unittest


class TestMeanSquaredError(unittest.TestCase):
    def setUp(self):
        self.test_cases = [
            (np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]), 0),
            (np.array([1, 2, 3, 4, 5]), np.array([2, 3, 4, 5, 6]), 1),
        ]

    def test_mean_squared_error(self):
        for y_true, y_pred, expected in self.test_cases:
            with self.subTest(y_pred=y_pred):
                np.testing.assert_almost_equal(mean_squared_error(y_true, y_pred), expected)


if __name__ == '__main__':
    unittest.main()
