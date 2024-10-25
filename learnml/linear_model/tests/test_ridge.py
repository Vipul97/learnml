from learnml.linear_model import Ridge
import numpy as np
import unittest


class TestRidge(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1], [2]])
        self.y = np.array([1, 2])
        self.lin_reg = Ridge(alpha=0.01)
        self.lin_reg.fit(self.X, self.y)

    def test_fit_predict(self):
        expected_coef = np.array([1])
        expected_intercept = np.array([0])
        expected_predictions = np.array([1, 2])

        np.testing.assert_almost_equal(self.lin_reg.coef_, expected_coef, 1)
        np.testing.assert_almost_equal(self.lin_reg.intercept_, expected_intercept, 1)
        np.testing.assert_almost_equal(self.lin_reg.predict(self.X), expected_predictions, 1)


if __name__ == '__main__':
    unittest.main()
