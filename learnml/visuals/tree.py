from graphviz import Digraph


def draw_tree(tree):
    def draw_tree_helper(subtree, t):
        root = list(subtree.keys())[0]
        t.node(root, root)

        for edge in subtree[root]:
            child = subtree[root][edge]

            if type(child) == dict:
                t.edge(root, list(child.keys())[0], label=edge)
                draw_tree_helper(child, t)

            else:
                t.node(edge + child, child, shape='box')
                t.edge(root, edge + child, label=edge)

    t = Digraph()
    draw_tree_helper(tree, t)
    t.render(view=True)
