from learnml.model_selection import train_test_split
import pandas as pd
import pandas.testing
import unittest


class Test(unittest.TestCase):
    def test_train_test_split(self):
        X = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        train_X, test_X = train_test_split(X, test_size=0.2, random_state=42)
        expected_train_X = pd.DataFrame([6, 1, 8, 3, 10, 5, 4, 7]).set_index(pd.Series([5, 0, 7, 2, 9, 4, 3, 6]))
        expected_test_X = pd.DataFrame([9, 2]).set_index(pd.Series([8, 1]))
        pandas.testing.assert_frame_equal(expected_train_X, train_X)
        pandas.testing.assert_frame_equal(expected_test_X, test_X)


if __name__ == '__main__':
    unittest.main()
