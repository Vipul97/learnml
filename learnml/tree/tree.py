import numpy as np
import pandas as pd


class DecisionTreeClassifier:
    def __init__(self, mode='ID3'):
        self.__mode = mode
        self.dataset_ = None
        self.features_ = None
        self.tree_ = None

    def fit(self, X, y):
        def label_counts(feature, dataset):
            counts = {}

            for label in dataset[feature]:
                counts[label] = counts.get(label, 0) + 1

            return counts

        def entropy(feature, dataset):
            feature_label_counts = label_counts(feature, dataset)
            e = 0.0

            for label in feature_label_counts:
                p = float(feature_label_counts[label]) / len(dataset)
                e += p * np.log2(p)

            return -e

        def rem(feature, dataset):
            feature_label_counts = label_counts(feature, dataset)
            r = 0.0

            for label in feature_label_counts:
                p = float(feature_label_counts[label]) / len(dataset)
                r += p * entropy(y.name, dataset[dataset[feature] == label])

            return r

        def info_gain(feature, dataset):
            return entropy(y.name, dataset) - rem(feature, dataset)

        def find_best_feature(features, dataset):
            max_info_gain = 0.0
            best_feature = None

            for feature in features:
                feature_info_gain = info_gain(feature, dataset)

                if self.__mode == 'C45':
                    feature_info_gain /= entropy(feature, dataset)

                if feature_info_gain > max_info_gain:
                    max_info_gain = feature_info_gain
                    best_feature = feature

            return best_feature

        def build_tree(features, dataset):
            if len(label_counts(y.name, dataset)) == 1:
                return list(label_counts(y.name, dataset).keys())[0]

            best_feature = find_best_feature(features, dataset)
            tree = {best_feature: {}}
            features.remove(best_feature)

            for label in label_counts(best_feature, dataset):
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
