from learnml.linear_model import Ridge
import numpy as np
import numpy.testing
import unittest


class TestRidge(unittest.TestCase):
    def test_fit_predict(self):
        X = np.array([[1], [2]])
        y = np.array([1, 2])

        lin_reg = Ridge(alpha=0.01)
        lin_reg.fit(X, y)

        numpy.testing.assert_almost_equal(np.array([1]), lin_reg.coef_, 1)
        numpy.testing.assert_almost_equal(np.array([0]), lin_reg.intercept_, 1)
        numpy.testing.assert_almost_equal(np.array([1, 2]), lin_reg.predict(X), 1)


if __name__ == '__main__':
    unittest.main()
