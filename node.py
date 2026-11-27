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
        self.value = None
        self.param_index = None
        self.len_split = None
        pass

    def split(self, x, y, max_depth, min_sample_leaf):
        print(self.depth)
        self.searchSplit(x, y, min_sample_leaf)

        self.node_p, x_p, y_p, self.node_n, x_m, y_m = self.applySplit(x, y)
        nb_leaf_p=y_p.shape[1]  
        if self.node_p.depth < max_depth and min_sample_leaf < nb_leaf_p:
            self.node_p.split(x_p, y_p, max_depth, min_sample_leaf)
        nb_leaf_m=y_p.shape[1]  
        if self.node_n.depth < max_depth and min_sample_leaf < nb_leaf_m:
            self.node_n.split(x_m, y_m, max_depth, min_sample_leaf)

        return

    def searchSplit(self, x, y, min_sample_leaf):
        min_mse = np.inf
        data = np.concatenate((x, y))
        nb_colonnes, nb_lignes = data.shape

        for i in range(0, nb_colonnes):
            feature = data[[i, -1], :]

            # Tri selon la première ligne
            indices = np.argsort(feature[0])

            # Réorganisation de toutes les lignes
            feature_sorted = feature[:, indices]
            if (nb_lignes>(min_sample_leaf*2-1)) :
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
            else : 
                features = feature_sorted[1,:]
                ## medianne ???
                self.value = np.average(features)

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
        y_p = data_p[-1:, :]

        x_m = data_m[:-1, :]
        y_m = data_m[-1:, :]

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

    def display_node(self, side = None):
        if self is not None:
            if self.limit is not None:
                print("   "*self.depth +  f"Depth : {self.depth} Limite : {self.limit}, ParamIndex : {self.param_index}, Side : {side}")
            elif self.value is not None:
               print("   "*self.depth +  f"Depth : {self.depth} Value = {round(self.value,2)} Side : {side}")
        if self.node_p is not None :
            self.node_p.display_node("Positive")
        if self.node_n is not None :
            self.node_n.display_node("Negative")
        
        return 