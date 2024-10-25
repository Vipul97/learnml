import numpy as np


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))
