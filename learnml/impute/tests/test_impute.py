from learnml.impute import SimpleImputer
import numpy as np
import numpy.testing
import pandas as pd
import unittest


class TestSimpleImputer(unittest.TestCase):
    def test_fit_transform(self):
        X = pd.DataFrame([1, 2, 2, None, 4, 5])
        expected_results = [np.array([[1.0], [2.0], [2.0], [2.8], [4.0], [5.0]]),
                            np.array([[1.0], [2.0], [2.0], [2.0], [4.0], [5.0]]),
                            np.array([[1.0], [2.0], [2.0], [2.0], [4.0], [5.0]]),
                            np.array([[1.0], [2.0], [2.0], [42.0], [4.0], [5.0]])]

        for i, strategy in enumerate(['mean', 'median', 'most_frequent', 'constant']):
            with self.subTest(strategy=strategy):
                imputer = SimpleImputer(strategy=strategy, fill_value=42)
                imputer.fit(X)
                numpy.testing.assert_equal(expected_results[i], imputer.transform(X))


if __name__ == '__main__':
    unittest.main()
