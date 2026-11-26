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

        self.searchSplit(x, y)

        node_p, x_p, y_p, node_n, x_n, y_n = self.applySplit(x, y)

        if node_p.depth < max_depth and min_sample_leaf < len(y_p):
            node_p.split(
                x_p,
            )
        if node_n.depth < max_depth and min_sample_leaf < len(y_p):
            node_n.split(x_n, y_n)

        return

    def searchSplit(x, y):

        # limit=
        # param_index=
        #
        pass

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
        