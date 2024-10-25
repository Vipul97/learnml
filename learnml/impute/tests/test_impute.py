from learnml.impute import SimpleImputer
import numpy as np
import pandas as pd
import unittest


class TestSimpleImputer(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({'values': [1, 2, 2, None, 4, 5]})
        self.expected_results = {
            'mean': np.array([[1.0], [2.0], [2.0], [2.8], [4.0], [5.0]]),
            'median': np.array([[1.0], [2.0], [2.0], [2.0], [4.0], [5.0]]),
            'most_frequent': np.array([[1.0], [2.0], [2.0], [2.0], [4.0], [5.0]]),
            'constant': np.array([[1.0], [2.0], [2.0], [42.0], [4.0], [5.0]])
        }


    def test_fit_transform(self):
        for strategy, expected in self.expected_results.items():
            with self.subTest(strategy=strategy):
                imputer = SimpleImputer(strategy=strategy, fill_value=42)
                imputer.fit(self.X)
                result = imputer.transform(self.X)
                np.testing.assert_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
