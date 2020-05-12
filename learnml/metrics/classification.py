import pandas as pd


def confusion_matrix(y_true, y_pred):
    return pd.crosstab(y_true, y_pred).to_numpy()
