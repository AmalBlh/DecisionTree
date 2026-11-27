from node import Node
import numpy as np

class DecisionTree:

    def __init__(self, max_depth, min_sample_leaf):
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.main_node = Node(0)
        return

    def fit(self, x, y):
        self.main_node = Node(0)
        if (self.max_depth>0) :
            self.main_node.split(x, y, self.max_depth, self.min_sample_leaf)
        return self

    def predict(self, x):
        y_pred = []
        l = x.shape[0]
        for i in range(l):
            value = self.main_node.predict(x[i])
            y_pred.append(value)

        return np.array(y_pred)

    def fit_predict(self, x, y):
        self.fit(x, y)
        self.predict(x)
        return


    def score(self, y, y_pred):
        y = np.array(y).flatten()
        y_pred = np.array(y_pred).flatten()

        n = len(y)
        somme = 0

        for i in range(n):
            somme += (y[i] - y_pred[i]) ** 2

        return somme / n


    def display_tree(self):
        self.main_node.display_node()
        return
