import numpy as np


class Node:

    def __init__(self, depth):
        self.node_p = None
        self.node_n = None
        self.depth = depth
        self.limit = None
        self.value = None
        self.param_index = None
        self.len_split = None
        

    def split(self, x, y, max_depth, min_sample_leaf):
        self.searchSplit(x, y, min_sample_leaf)
        if self.len_split is not None:
            self.node_p, x_p, y_p, self.node_n, x_m, y_m = self.applySplit(x, y)
            nb_leaf_p = y_p.shape[0]
            if self.node_p.depth < max_depth and min_sample_leaf < nb_leaf_p:
                self.node_p.split(x_p, y_p, max_depth, min_sample_leaf)
            else:
                self.node_p.compute_value(x_p, y_p)
            nb_leaf_m = y_m.shape[0]
            if self.node_n.depth < max_depth and min_sample_leaf < nb_leaf_m:
                self.node_n.split(x_m, y_m, max_depth, min_sample_leaf)
            else:
                self.node_n.compute_value(x_m, y_m)
        else:
            self.compute_value(x, y)

        return

    def searchSplit(self, x, y, min_sample_leaf):
        min_mse = np.inf
        data = self._combine_features_target(x, y)
        n_samples, n_columns = data.shape
        n_features = n_columns - 1
        targets = data[:, -1]

        for feature_index in range(n_features):
            indices = np.argsort(data[:, feature_index])
            feature_values = data[indices, feature_index]
            sorted_targets = targets[indices]

            for split_index in range(min_sample_leaf, n_samples - min_sample_leaf + 1):
                if feature_values[split_index - 1] >= feature_values[split_index]:
                    continue

                left_targets = sorted_targets[:split_index]
                right_targets = sorted_targets[split_index:]
                left_mean = np.mean(left_targets)
                right_mean = np.mean(right_targets)
                mse = (
                    np.sum((left_targets - left_mean) ** 2)
                    + np.sum((right_targets - right_mean) ** 2)
                ) / n_samples

                if mse < min_mse:
                    min_mse = mse
                    self.limit = (
                        feature_values[split_index - 1]
                        + (feature_values[split_index] - feature_values[split_index - 1]) / 2
                    )
                    self.param_index = feature_index
                    self.len_split = split_index

        return

    def applySplit(self, x, y):
        if self.len_split is not None:
            data = self._combine_features_target(x, y)
            indices = np.argsort(data[:, self.param_index])
            data_sorted = data[indices]
            limit = self.len_split

            data_p = data_sorted[limit:]
            data_m = data_sorted[:limit]
            x_p, y_p = data_p[:, :-1], data_p[:, -1]
            x_m, y_m = data_m[:, :-1], data_m[:, -1]

            node_p = Node(self.depth + 1)
            node_n = Node(self.depth + 1)

            return node_p, x_p, y_p, node_n, x_m, y_m
        return None

    def compute_value(self, x, y):
        self._combine_features_target(x, y)
        self.value = np.mean(y)
        return

    @staticmethod
    def _combine_features_target(x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        if x.ndim != 2:
            raise ValueError("x must have shape (n_samples, n_features)")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim != 2 or y.shape[1] != 1:
            raise ValueError("y must have shape (n_samples,) or (n_samples, 1)")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must contain the same number of samples")

        return np.concatenate((x, y), axis=1)


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
                print("   "*self.depth +  f"Depth : {self.depth} Limite : {round(self.limit,2)}, ParamIndex : {self.param_index}, Side : {side}")
            elif self.value is not None:
               print("   "*self.depth +  f"Depth : {self.depth} Value = {round(self.value,2)} Side : {side}")
        if self.node_p is not None :
            self.node_p.display_node("Positive")
        if self.node_n is not None :
            self.node_n.display_node("Negative")
        
        return 