import node


class DecisionTree:
    # score
    # main_node

    def __init__(self, min_simple_leaf, max_depth):
        self.max_depth = max_depth
        self.min_simple_leaf = min_simple_leaf
        self.main_node = node(0)
        return

    def fit(self, x, y):
        self.main_node.split(x, y, self.max_depth, self.min_sample_leaf)
        return self.main_node

    def predict(x):
        pass
        # return y(array)

    def fit_predict(x, y):
        pass
        # a verifier

    def scoring():
        pass
        # retrn score (MSE)
