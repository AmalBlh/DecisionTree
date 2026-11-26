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
        self.limit = None
        self.param_index = None
        self.len_split = None
        pass

    def split(self, x, y, max_depth, min_sample_leaf):

        self.searchSplit(x, y, min_sample_leaf)

        node_p, x_p, y_p, node_n, x_m, y_m = self.applySplit(x, y)

        if node_p.depth < max_depth and min_sample_leaf < len(y_p):
            node_p.split(x_p, y_p, max_depth, min_sample_leaf)
        if node_n.depth < max_depth and min_sample_leaf < len(y_m):
            node_n.split(x_m, y_m, max_depth, min_sample_leaf)

        return

    def searchSplit(self, x, y, min_sample_leaf):
        min_mse = np.inf

        nb_colonnes, nb_lignes = x.shape

        for i in range(0, nb_colonnes):
            feature = x[[i, -1], :]

            # Tri selon la première ligne
            indices = np.argsort(feature[0])

            # Réorganisation de toutes les lignes
            feature_sorted = feature[:, indices]

            for limit in range(min_sample_leaf, nb_lignes - min_sample_leaf):

                feature_m = feature_sorted[1, 0:limit]
                feature_p = feature_sorted[1, limit + 1 : nb_lignes]
                average_m = np.average(feature_m)
                average_p = np.average(feature_p)

                mse_m = np.mean((feature_m - average_m) ** 2)
                mse_p = np.mean((feature_p - average_p) ** 2)
                mse = (mse_m * len(feature_m) + mse_p * len(feature_p)) / nb_lignes
                if mse < min_mse:
                    self.value = feature[0, i]
                    mse = min_mse
                    self.limit = limit
                    self.param_index = i
                    self.len_split = len(feature_m)

        return

    def applySplit(self, x, y):
        data = np.concatenate((x, y))
        nb_lignes, nb_colonnes = data.shape
        # data_sorted = data[np.argsort(data[:, self.param_index], axis=0)]
        # depth+1

        # Tri selon la première ligne
        indices = np.argsort(data[self.param_index])

        # Réorganisation de toutes les lignes
        data_sorted = data[:, indices]

        limit = self.len_split

        data_p = data_sorted[:, :limit]
        data_m = data_sorted[:, limit:]

        # Separar X e y (suponiendo que la última fila contiene y)
        x_p = data_p[:-1, :]
        y_p = [data_p[-1, :].T]

        x_m = data_m[:-1, :]
        y_m = [data_m[-1, :].T]

        node_p = Node(self.depth + 1)
        node_n = Node(self.depth + 1)

        return node_p, x_p, y_p, node_n, x_m, y_m

    def predict(self, x):

        if self.node_p is None and self.node_n is None:
            return self.value

        param = self.param_index

        if x[param] > self.limit:
            return self.node_p.predict(x)
        else:
            return self.node_n.predict(x)
