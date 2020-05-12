from learnml.preprocessing import StandardScaler
import numpy as np
import numpy.testing
import unittest


class TestStandardScaler(unittest.TestCase):
    def test_fit_transform(self):
        X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

        scaler = StandardScaler()

        scaler.fit(X)

        numpy.testing.assert_array_almost_equal(np.array([3]), scaler.mean_)
        numpy.testing.assert_array_almost_equal(np.array([1.41421356]), scaler.scale_)


if __name__ == '__main__':
    unittest.main()
