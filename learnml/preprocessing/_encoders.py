import numpy as np
import pandas as pd


class OneHotEncoder:
    def __init__(self):
        self.categories_ = []

    def fit(self, data):
        for column in data.columns:
            self.categories_.append(np.array(data[column].sort_values().unique().astype(object)))

    def transform(self, data):
        return pd.get_dummies(data).to_numpy()


class OrdinalEncoder:
    def __init__(self):
        self.categories_ = []

    def fit(self, data):
        for column in data.columns:
            self.categories_.append(np.array(data[column].sort_values().unique().astype(object)))

    def transform(self, data):
        for column in data.columns:
            data[column] = data[column].astype('category')
            data[column] = data[column].cat.codes

        return data.to_numpy()
