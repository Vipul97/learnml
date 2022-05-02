from learnml.linear_model import LinearRegression
import numpy as np
import numpy.testing
import unittest


class TestLinearRegression(unittest.TestCase):
    def test_fit_predict(self):
        X = np.array([[1], [2]])
        y = np.array([1, 2])

        lin_reg = LinearRegression()
        lin_reg.fit(X, y)

        numpy.testing.assert_almost_equal(np.array([1]), lin_reg.coef_)
        numpy.testing.assert_almost_equal(np.array([0]), lin_reg.intercept_)
        numpy.testing.assert_almost_equal(np.array([1, 2]), lin_reg.predict(X))


if __name__ == '__main__':
    unittest.main()
