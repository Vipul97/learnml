import numpy as np
import pandas as pd


class OneHotEncoder:
    def __init__(self):
        self.categories_ = {}

    def fit(self, data):
        self.categories_ = {col: data[col].unique() for col in data.columns}

    def transform(self, data):
        return pd.get_dummies(data).to_numpy()


class OrdinalEncoder:
    def __init__(self):
        self.categories_ = {}

    def fit(self, data):
        self.categories_ = {col: data[col].unique() for col in data.columns}

    def transform(self, data):
        for column in data.columns:
            data[column] = data[column].astype('category').cat.codes

        return data.to_numpy()
