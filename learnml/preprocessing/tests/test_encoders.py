from learnml.preprocessing import OrdinalEncoder
from learnml.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import unittest


class BaseEncoderTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv('learnml/preprocessing/tests/test_data.csv')


class TestOneHotEncoder(BaseEncoderTest):
    def test_fit_transform(self):
        onehot_encoder = OneHotEncoder()
        onehot_encoder.fit(self.data)

        expected_output = np.array([
            [False, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
            [True, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [True, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [False, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
            [False, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [True, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [True, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
        ])

        np.testing.assert_equal(onehot_encoder.transform(self.data), expected_output)


class TestOrdinalEncoder(BaseEncoderTest):
    def test_fit_transform(self):
        ordinal_encoder = OrdinalEncoder()
        ordinal_encoder.fit(self.data)

        expected_output = np.array([
            [0, 2, 0, 0],
            [1, 1, 2, 2],
            [1, 2, 3, 2],
            [0, 2, 3, 0],
            [0, 0, 0, 1],
            [1, 2, 1, 1],
            [1, 2, 0, 0]
        ])

        np.testing.assert_equal(ordinal_encoder.transform(self.data), expected_output)


if __name__ == '__main__':
    unittest.main()
