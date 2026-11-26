import numpy as np


class Node:

    # limit
    # param_index

    # depth
    # value

    def __init__(self, depth):
        self.node_p = None
        self.node_n = None
        self.depth = depth
        pass

    def split(self, x, y, max_depth, min_sample_leaf):

        self.searchSplit(x, y, min_sample_leaf)

        node_p, x_p, y_p, node_n, x_n, y_n = self.applySplit(x, y)

        if node_p.depth < max_depth and min_sample_leaf < len(y_p):
            node_p.split(
                x_p,
            )
        if node_n.depth < max_depth and min_sample_leaf < len(y_p):
            node_n.split(x_n, y_n)

        return

    def searchSplit(self, x, y, min_sample_leaf):
        min_mse = np.inf
        min_limit = None
        feature = 0
        value = None
        for i in range(0, len(x)):
            feature = x[:, [i, -1]]

            indices = np.argsort(feature[:, i])

            for limit in range(min_sample_leaf, len(indices) - min_sample_leaf):

                feature_m = feature[i, [0, limit]]
                feature_p = feature[i, [limit + 1, len(feature)]]
                average_m = np.average(feature_m)
                average_p = np.average(feature_p)

                mse_m = ((feature_m - average_m) ** 2).mean(feature_m)
                mse_p = ((feature_p - average_p) ** 2).mean(feature_p)
                mse = (mse_m * len(feature_m) + mse_p * len(feature_p)) / len(feature)
                if mse < min_mse:
                    value = feature[i, 0]
                    mse = min_mse
                    min_limit = limit
                    feature = i

        self.limit = min_limit
        self.param_index = feature
        self.value = value

        return

    def applySplit(x, y):

        # depth+1
        # return node_p, x_p, y_p, node_n, x_n, y_n
        pass


    def predict(self, x):

        param = self.param_index
        if(x[param] > self.limit):
            self.node_p.predict(x)
        else :
            self.node_n.predict(x)

        if self.node_n is None :
            return self.node_n.value 
        else :
            if self.node_p is None :
                return self.node_p.value       
        