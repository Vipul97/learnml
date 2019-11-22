import csv
import numpy as np
from graphviz import Digraph

ID3 = 0
C45 = 1


def count(feature, dataset):
    counts = {}

    for label in dataset[:, features.index(feature)]:
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
        r += p * entropy(features[-1], dataset[dataset[:, features.index(feature)] == label])

    return r


def info_gain(feature, dataset):
    return entropy(features[-1], dataset) - rem(feature, dataset)


def find_best_feature(features, dataset, mode):
    max_info_gain = 0.0
    best_feature = ""

    for feature in features:
        curr_info_gain = info_gain(feature, dataset)

        if mode == C45:
            curr_info_gain /= entropy(feature, dataset)

        if curr_info_gain > max_info_gain:
            max_info_gain = curr_info_gain
            best_feature = feature

    return best_feature


def build_tree(f, dataset, mode):
    if len(count(features[-1], dataset)) == 1:
        return dataset[0, -1]

    best_feature = find_best_feature(f, dataset, mode)
    tree = {best_feature: {}}
    f.remove(best_feature)

    for val in count(best_feature, dataset):
        tree[best_feature][val] = build_tree(f, dataset[dataset[:, features.index(best_feature)] == val], mode)

    return tree


def draw_tree(tree):
    for root in tree.keys():
        t.node(root, root)
        for edge in tree[root]:
            child = tree[root][edge]

            if type(child) == dict:
                for ch in child.keys():
                    t.edge(root, ch, label=edge)
                    draw_tree(child)
            else:
                t.node(root + child, child, shape='box')
                t.edge(root, root + child, label=edge)


with open('data.csv', 'r') as csvfile:
    csv = list(csv.reader(csvfile))
    features, dataset = csv[0], np.array(csv[1:])


for mode in [ID3, C45]:
    t = Digraph()
    draw_tree(build_tree(features[:-1], dataset, mode))

    if mode == ID3:
        t.render('ID3 Decision Tree', view=True)
    else:
        t.render('C4.5 Decision Tree', view=True)
