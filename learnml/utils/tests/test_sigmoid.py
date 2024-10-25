from learnml.utils import sigmoid
import numpy as np
import unittest


class Test(unittest.TestCase):
    def test_sigmoid(self):
        np.testing.assert_(sigmoid(-1) < 0.5)
        np.testing.assert_equal(sigmoid(0), 0.5)
        np.testing.assert_(sigmoid(1) > 0.5)


if __name__ == '__main__':
    unittest.main()
