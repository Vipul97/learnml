from learnml.model_selection import train_test_split
import pandas as pd
import pandas.testing
import unittest


class TestTrainTestSplit(unittest.TestCase):
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    def setUp(self):
        self.X = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_train_test_split(self):
        train_X, test_X = train_test_split(self.X, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE)

        expected_train_X = pd.DataFrame([6, 1, 8, 3, 10, 5, 4, 7]).set_index(pd.Series([5, 0, 7, 2, 9, 4, 3, 6]))
        expected_test_X = pd.DataFrame([9, 2]).set_index(pd.Series([8, 1]))

        pandas.testing.assert_frame_equal(train_X, expected_train_X)
        pandas.testing.assert_frame_equal(test_X, expected_test_X)


if __name__ == '__main__':
    unittest.main()
