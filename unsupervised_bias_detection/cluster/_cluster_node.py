import itertools
from sklearn.base import ClusterMixin
from typing import Self

class ClusterNode:
    _id_counter = itertools.count()

    def __init__(self, neg_std: float, score: float):
        """
        Initialize a node in the cluster tree.
        
        Parameters
        ----------
        label : int
            The cluster label for this node (required as all nodes start as leaves)
        """
        self.id = next(self._id_counter)
        self.neg_std = neg_std
        self.score = score
        # The label is set to the id when the node is a leaf
        # and is set to None when the node is split
        self.label = self.id
        self.clustering_model = None
        self.children = []
    
    @property
    def is_leaf(self):
        return len(self.children) == 0
    
    def __lt__(self, other: Self):
        return self.neg_std < other.neg_std or (self.neg_std == other.neg_std and self.id < other.id)
    
    def split(self, clustering_model: ClusterMixin, children: list[Self]):
        """
        Split this node by setting its clustering model and adding children.
        
        This converts the node to an internal node and removes its label
        
        Parameters
        ----------
        clustering_model : ClusterMixin
            The clustering model used to split this node
        children : list of ClusterNode
            The child nodes resulting from the split
        """
        self.label = None
        self.clustering_model = clustering_model
        self.children = children
    
    def get_leaves(self) -> list[Self]:
        """
        Get all leaf nodes in the subtree rooted at this node.
        
        Returns
        -------
        list of ClusterNode
            All leaf nodes in the subtree
        """
        if not self.children:
            return [self]
        
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves