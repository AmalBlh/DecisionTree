from node import Node


class DecisionTree:
    # score
    # main_node
    # min_sample_leaf = None
    # max_depth = None

    def __init__(self, min_sample_leaf, max_depth):
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.main_node = Node(0)
        return

    def fit(self, x, y):
        self.main_node.split(x, y, self.max_depth, self.min_sample_leaf)
        return self.main_node

    def predict(self, x):
        y = []

        for i in range(x):
            value = self.main_node.predict(x[i])
            y.append(value)

        return y

    def fit_predict(x, y):
        pass
        # a verifier

    def score(y, y_pred):
        n = len(y)
        somme = 0
        for i in range(n):
            somme += (y[i] - y_pred[i]) ** 2

        return somme / n

    def display_tree(self, main_node):
        if main_node is not None:
            print(f"Limite : {main_node.limit}, Index : {main_node.param_index}")

            self.display_tree(main_node.node_p)
            self.display_tree(main_node.node_n)
