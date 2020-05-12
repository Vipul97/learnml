import numpy as np
from graphviz import Digraph


class DecisionTreeClassifier:
    def __init__(self, mode='ID3'):
        self.__mode = mode
        self.dataset = None
        self.features = None
        self.t = Digraph()

    def fit(self, features, dataset):
        def count(feature, dataset):
            counts = {}

            for label in dataset[feature]:
                if label not in counts:
                    counts[label] = 0
                counts[label] += 1

            return counts

        def entropy(feature, dataset):
            counts_of_label = count(feature, dataset)
            e = 0.0

            for label in counts_of_label:
                p = float(counts_of_label[label]) / len(dataset)
                e += p * np.log2(p)

            return -e

        def rem(feature, dataset):
            counts_of_label = count(feature, dataset)
            r = 0.0

            for label in counts_of_label:
                p = float(counts_of_label[label]) / len(dataset)
                r += p * entropy(features[-1], dataset[dataset[feature] == label])

            return r

        def info_gain(feature, dataset):
            return entropy(features[-1], dataset) - rem(feature, dataset)

        def find_best_feature(features, dataset):
            max_info_gain = 0.0
            best_feature = ''

            for feature in features:
                curr_info_gain = info_gain(feature, dataset)

                if self.__mode == 'C45':
                    curr_info_gain /= entropy(feature, dataset)

                if curr_info_gain > max_info_gain:
                    max_info_gain = curr_info_gain
                    best_feature = feature

            return best_feature

        def build_tree(f, dataset):
            if len(count(features[-1], dataset)) == 1:
                return list(count(features[-1], dataset).keys())[0]

            best_feature = find_best_feature(f, dataset)
            tree = {best_feature: {}}
            f.remove(best_feature)

            for val in count(best_feature, dataset):
                tree[best_feature][val] = build_tree(f, dataset.loc[dataset[best_feature] == val])

            return tree

        self.features = features
        self.dataset = dataset

        return build_tree(self.features[:-1], self.dataset)
