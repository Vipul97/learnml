import numpy as np
import pandas as pd


class DecisionTreeClassifier:
    def __init__(self, mode='ID3'):
        self.__mode = mode
        self.dataset_ = None
        self.features_ = None
        self.tree_ = None

    def fit(self, X, y):
        def entropy(feature, dataset):
            e = 0.0

            for count in dataset[feature].value_counts():
                p = float(count) / len(dataset)
                e += p * np.log2(p)

            return -e

        def rem(feature, dataset):
            r = 0.0

            for label, count in dataset[feature].value_counts().items():
                p = float(count) / len(dataset)
                r += p * entropy(y.name, dataset[dataset[feature] == label])

            return r

        def info_gain(feature, dataset):
            return entropy(y.name, dataset) - rem(feature, dataset)

        def find_best_feature(features, dataset):
            max_info_gain = 0.0
            best_feature = None

            for feature in features:
                feature_info_gain = info_gain(feature, dataset)

                if self.__mode == 'C4.5':
                    feature_info_gain /= entropy(feature, dataset)

                if max_info_gain < feature_info_gain:
                    max_info_gain = feature_info_gain
                    best_feature = feature

            return best_feature

        def build_tree(features, dataset):
            target_label_counts = dataset[y.name].value_counts()
            if len(target_label_counts) == 1:
                return target_label_counts.index[0]

            best_feature = find_best_feature(features, dataset)
            tree = {best_feature: {}}
            features.remove(best_feature)

            for label in dataset[best_feature].value_counts().index:
                tree[best_feature][label] = build_tree(features, dataset.loc[dataset[best_feature] == label])

            return tree

        self.dataset_ = pd.concat([X, y], axis=1)
        self.features_ = list(X.columns)
        self.tree_ = build_tree(self.features_, self.dataset_)

    def predict(self, X):
        y = []

        for x in X.to_dict('records'):
            class_ = self.tree_

            while type(class_) == dict:
                feature = list(class_.keys())[0]
                class_ = class_[feature][x[feature]]

            y.append(class_)

        return y
