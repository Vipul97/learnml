from graphviz import Digraph


def draw_tree(tree):
    def draw_tree_helper(subtree, t):
        root = next(iter(subtree))
        t.node(root, root)

        for edge, child in subtree[root].items():
            if isinstance(child, dict):
                t.edge(root, next(iter(child)), label=edge)
                draw_tree_helper(child, t)
            else:
                t.node(f'{edge}{child}', child, shape='box')
                t.edge(root, f'{edge}{child}', label=edge)

    t = Digraph()
    draw_tree_helper(tree, t)
    t.render(view=True)
