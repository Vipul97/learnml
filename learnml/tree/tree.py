import numpy as np
import pandas as pd


class DecisionTreeClassifier:
    def __init__(self, mode='ID3'):
        self.mode = mode
        self.dataset = None
        self.features = None
        self.tree = None

    def _entropy(self, feature, dataset):
        probabilities = dataset[feature].value_counts(normalize=True)
        return -np.sum(probabilities * np.log2(probabilities))

    def _rem(self, feature, dataset, target_name):
        total_count = len(dataset)
        return np.sum(
            (count / total_count) * self._entropy(target_name, dataset[dataset[feature] == label])
            for label, count in dataset[feature].value_counts().items()
        )

    def _info_gain(self, feature, dataset, target_name):
        return self._entropy(target_name, dataset) - self._rem(feature, dataset, target_name)

    def _find_best_feature(self, features, dataset, target_name):
        gains = {feature: self._info_gain(feature, dataset, target_name) for feature in features}

        if self.mode == 'C4.5':
            total_entropy = self._entropy(target_name, dataset)
            gains = {feature: gain / total_entropy for feature, gain in gains.items()}

        return max(gains, key=gains.get)

    def _build_tree(self, features, dataset, target_name):
        target_counts = dataset[target_name].value_counts()

        if len(target_counts) == 1:
            return target_counts.index[0]

        best_feature = self._find_best_feature(features, dataset, target_name)
        tree = {best_feature: {}}
        remaining_features = features.copy()
        remaining_features.remove(best_feature)

        for label in dataset[best_feature].unique():
            subset = dataset[dataset[best_feature] == label]
            tree[best_feature][label] = self._build_tree(remaining_features, subset, target_name)

        return tree

    def fit(self, X, y):
        self.dataset = pd.concat([X, y], axis=1)
        self.features = list(X.columns)
        self.tree = self._build_tree(self.features, self.dataset, y.name)

    def _traverse_tree(self, tree, record):
        while isinstance(tree, dict):
            feature = next(iter(tree))
            tree = tree[feature][record[feature]]
        return tree

    def predict(self, X):
        return [self._traverse_tree(self.tree, x) for x in X.to_dict('records')]
