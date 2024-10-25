from learnml.linear_model import LinearRegression
import numpy as np
import unittest


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1], [2]])
        self.y = np.array([1, 2])
        self.lin_reg = LinearRegression()
        self.lin_reg.fit(self.X, self.y)

    def test_fit_predict(self):
        expected_coef = np.array([1])
        expected_intercept = np.array([0])
        expected_prediction = np.array([1, 2])

        np.testing.assert_almost_equal(self.lin_reg.coef_, expected_coef)
        np.testing.assert_almost_equal(self.lin_reg.intercept_, expected_intercept)
        np.testing.assert_almost_equal(self.lin_reg.predict(self.X), expected_prediction)


if __name__ == '__main__':
    unittest.main()
