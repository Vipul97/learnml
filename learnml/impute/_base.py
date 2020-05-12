class SimpleImputer:
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, data):
        if self.strategy == 'mean':
            self.statistics_ = data.mean()
        elif self.strategy == 'median':
            self.statistics_ = data.median()
        elif self.strategy == 'most_frequent':
            self.statistics_ = data.mode().iloc[0]
        elif self.strategy == 'constant':
            self.statistics_ = self.fill_value

    def transform(self, data):
        return data.fillna(self.statistics_).to_numpy()
