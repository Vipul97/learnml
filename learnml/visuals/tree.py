from graphviz import Digraph


def draw_tree(tree):
    t = Digraph()

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
                t.edge(root, root + child, label=str(edge))

    t.render(view=True)
