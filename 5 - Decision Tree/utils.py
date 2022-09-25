import numpy as np

class Node(object):
    def __init__(
        self, 
        ids: list = None, 
        children: list = [], 
        entropy: float = 0.0, 
        depth: int = 0
    ):
        """
        Represent a node in tree
        
        Args:
            ids: (int) index of data in this node
            children: (list) list of its child nodes
            entropy: (float)
            depth: (int) distance from this node to root node
        """
        
        self.ids = ids  
        self.children = children 
        self.entropy = entropy
        self.depth = depth      
        self.split_attribute = None     # which attribute is chosen, it non-leaf
        self.order = None     # order of values of split_attribute in children
        self.label = None     # label of node if it is a leaf

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label
        
    def __str__(self):
        information = (
            f'Index of node: {self.ids}\n'
            f'Children: {self.children}\n'
            f'Entropy: {self.entropy}\n'
            f'Depth: {self.depth}'
        )
        return information
    

class DecisionTreeID3(object):
    def __init__(
        self, 
        max_depth: int = 10, 
        min_samples_split: int = 2, 
        min_gain: float = 1e-4
    ):
        """
        Parameters of model
        
        Args:
            max_depth: (int) the maximum depth of the tree
            min_samples_split: (int) the minimum number of samples required to split an internal node
            min_gain: (float) the minimum amount the Information Gain must increase for the tree to continue growing
        """
        self.root = None
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split 
        self.N = 0
        self.min_gain = min_gain
    
    def fit(self, data, target):
        """Function to fit model"""
        self.N = data.count()[0]
        self.data = data 
        self.attributes = list(data)
        self.target = target 
        self.labels = target.unique()
        
        ids = range(self.N)
        self.root = Node(ids=ids, entropy=self._entropy(self.target), depth=0)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if not node.children:     # leaf node
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)
                
    def _entropy(self, target):
        """
        Calculate entropy of the label set S

        Args:
            target: (ndarray) labels in set S
        Return:
            en: (float) entropy value of input 'target'
        """
        _, freqs = np.unique(target, return_counts=True)
        probs = freqs / float(freqs.sum())
        en = - np.sum(probs * np.log(probs))
        return en
    
    def _information_gain(self, node, T):
        """
        Calculate information gain of a tree after spliting.

        Args:
            node: (Node) (parent) node, node need to be splitted
            T: (list) contain all indexs of each children
            target: (ndarray) labels of set S
        Return:
            ig: information gain of node T after being splitted by attribute A
        """
        n = len(node.ids)
        H_SA = 0
        for ids in T:
            p = len(ids) / n
            H_SA += p * self._entropy(self.target[ids])
        ig = node.entropy - H_SA
        return ig

    def _set_label(self, node):
        """Find label for a node if it is a leaf, simply chose by major voting"""
        node.set_label(self.target[node.ids].mode()[0])     # most frequent label
    
    def _split(self, node):
        """
        Function for making best split

        Args:
            node: (Node) a node of tree
            data: (DataFrame) original data from input
            target: (ndarray) labels corresponds to each row in data
        Return:
            child_nodes: (list) contain all childrens of parent node
        """
        ids = node.ids 
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
            if len(values) == 1: continue     # entropy = 0
            splits = []
            for val in values: 
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append(sub_ids)
            # don't split if a node has too small number of points
            if min(map(len, splits)) < self.min_samples_split: continue
            # information gain
            gain = self._information_gain(node, splits)
            if gain < self.min_gain: continue     # stop if small gain 
            if gain > best_gain:
                best_gain = gain 
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order)
        child_nodes = [Node(ids = split, entropy = self._entropy(self.target[split]), depth = node.depth+1) 
                       for split in best_splits]
        return child_nodes

    def predict(self, new_data):
        """Predict data from input"""
        n_points = new_data.count()[0]
        labels = [None] * n_points
        for i in range(n_points):
            x = new_data.iloc[i, :]     # one point 
            # start from root and recursively travel if not meet a leaf 
            node = self.root
            while node.children: 
                node = node.children[node.order.index(x[node.split_attribute])]
            labels[i] = node.label
            
        return np.array(labels)
