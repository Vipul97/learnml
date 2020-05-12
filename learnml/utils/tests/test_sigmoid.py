from learnml.utils import sigmoid
import unittest


class Test(unittest.TestCase):
    def test_sigmoid(self):
        self.assertTrue(sigmoid(-1) < 0.5)
        self.assertEqual(0.5, sigmoid(0))
        self.assertTrue(sigmoid(1) > 0.5)


if __name__ == '__main__':
    unittest.main()
