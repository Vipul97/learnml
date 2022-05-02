from learnml.preprocessing import OrdinalEncoder
from learnml.preprocessing import OneHotEncoder
import numpy as np
import numpy.testing
import pandas as pd
import unittest


class TestOneHotEncoder(unittest.TestCase):
    def test_fit_transform(self):
        data = pd.read_csv('learnml/preprocessing/tests/test_data.csv')

        onehot_encoder = OneHotEncoder()
        onehot_encoder.fit(data)

        numpy.testing.assert_equal(np.array(
            [[False, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0], [True, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
             [True, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], [False, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
             [False, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0], [True, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
             [True, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0]]), onehot_encoder.transform(data))


class TestOrdinalEncoder(unittest.TestCase):
    def test_fit_transform(self):
        data = pd.read_csv('learnml/preprocessing/tests/test_data.csv')

        ordinal_encoder = OrdinalEncoder()
        ordinal_encoder.fit(data)

        numpy.testing.assert_equal(np.array(
            [[0, 2, 0, 0], [1, 1, 2, 2], [1, 2, 3, 2], [0, 2, 3, 0], [0, 0, 0, 1], [1, 2, 1, 1], [1, 2, 0, 0]]),
            ordinal_encoder.transform(data))


if __name__ == '__main__':
    unittest.main()
