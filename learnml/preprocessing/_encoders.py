import numpy as np


class OrdinalEncoder:
    def __init__(self):
        self.categories_ = []

    def fit(self, X):
        for column in X.columns:
            self.categories_.append(np.array(X[column].sort_values().unique().astype(object)))

    def transform(self, X):
        for column in X.columns:
            X[column] = X[column].astype("category")
            X[column] = X[column].cat.codes

        return X
