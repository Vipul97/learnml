from learnml.preprocessing import StandardScaler
import numpy as np
import unittest


class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        self.X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

    def test_fit_transform(self):
        scaler = StandardScaler()

        scaler.fit(self.X)

        np.testing.assert_array_almost_equal(scaler.mean_, np.array([3]))
        np.testing.assert_array_almost_equal(scaler.scale_, np.array([1.41421356]))


if __name__ == '__main__':
    unittest.main()
