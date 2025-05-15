from ._cluster_node import ClusterNode
from collections import deque
import heapq
from numbers import Integral
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import validate_data
from typing import Any, Type


class BiasAwareHierarchicalClustering(BaseEstimator, ClusterMixin):
    """
    TODO: Add docstring.

    References
    ----------
    .. [1] J. Misztal-Radecka, B. Indurkhya, "Bias-Aware Hierarchical Clustering for detecting the discriminated
           groups of users in recommendation systems", Information Processing & Management, vol. 58, no. 3, May. 2021.
    """

    _parameter_constraints: dict = {
        "bahc_max_iter": [Interval(Integral, 1, None, closed="left")],
        "bahc_min_cluster_size": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        clustering_cls: Type[ClusterMixin],
        bahc_max_iter: int,
        bahc_min_cluster_size: int,
        **clustering_params: Any,
    ):
        self.clustering_cls = clustering_cls
        self.bahc_max_iter = bahc_max_iter
        self.bahc_min_cluster_size = bahc_min_cluster_size
        self.clustering_params = clustering_params

    def fit(self, X, y):
        """Compute bias-aware hierarchical clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : array-like of shape (n_samples)
            Metric values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = validate_data(
            self,
            X,
            y,
            reset=False,
            accept_large_sparse=False,
            order="C",
        )
        n_samples, _ = X.shape
        # We start with all samples being in a single cluster
        self.n_clusters_ = 1
        # We assign all samples a label of zero
        labels = np.zeros(n_samples, dtype=np.uint32)
        leaves = []
        std = np.std(y)
        score = 0
        root = ClusterNode(0, -std, score)
        self.cluster_tree_ = root
        n_nodes = 1
        # The entire dataset has a discrimination score of zero
        heap = [root]
        for _ in range(self.bahc_max_iter):
            if not heap:
                # If the heap is empty we stop iterating
                break
            # Take the cluster with the highest standard deviation of metric y
            node = heapq.heappop(heap)
            label = node.label
            score = node.score
            cluster_indices = np.nonzero(labels == label)[0]
            X_cluster = X[cluster_indices]

            clustering_model = self.clustering_cls(**self.clustering_params)
            cluster_labels = clustering_model.fit_predict(X_cluster)

            if hasattr(clustering_model, "n_clusters_"):
                n_children = clustering_model.n_clusters_
            else:
                n_children = len(np.unique(cluster_labels))
            
            # We first check if all child clusters meet the minimum size requirement
            valid_split = True
            children_indices = []
            for i in range(n_children):
                child_indices = cluster_indices[np.nonzero(cluster_labels == i)[0]]
                if len(child_indices) >= self.bahc_min_cluster_size:
                    children_indices.append(child_indices)
                else:
                    leaves.append(node)
                    valid_split = False
                    break
                        
            # If all children clusters are of sufficient size, we check if the score of each child cluster is greater than or equal to the current score
            if valid_split:
                child_scores = []
                for child_indices in children_indices:
                    y_cluster = y[child_indices]
                    complement_mask = np.ones(n_samples, dtype=bool)
                    complement_mask[child_indices] = False
                    y_complement = y[complement_mask]
                    child_score = np.mean(y_complement) - np.mean(y_cluster)
                    if child_score >= score:
                        child_scores.append(child_score)
                    else:
                        leaves.append(node)
                        valid_split = False
                        break
            
            # If the split is valid, we create the children nodes and split the current node
            if valid_split:
                children = []
                for i in range(n_children):
                    child_indices = children_indices[i]
                    child_std = np.std(y[child_indices])
                    child_score = child_scores[i]
                    # heapq implements min-heap
                    # so we have to negate std before pushing
                    child_node = ClusterNode(n_nodes, -child_std, child_score)
                    children.append(child_node)
                    n_nodes += 1
                node.split(clustering_model, children)
                self.n_clusters_ += n_children - 1
        
        leaves.extend(heap)
        leaf_scores = np.array([leaf.score for leaf in leaves])
        # We sort clusters by decreasing scores
        sorted_indices = np.argsort(-leaf_scores)
        self.scores_ = leaf_scores[sorted_indices]
        leaf_labels = np.array([leaf.label for leaf in leaves])
        leaf_labels = leaf_labels[sorted_indices]
        label_mapping = np.zeros(self.n_clusters_, dtype=np.uint32)
        label_mapping[leaf_labels] = np.arange(self.n_clusters_, dtype=np.uint32)
        self.labels_ = label_mapping[labels]
        for leaf in leaves:
            leaf.label = label_mapping[leaf.label]
        return self
    
    def predict(self, X):
        """Predict the cluster labels for the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        # TODO: Assert that fit has been called
        # TODO: Assert that X has the same number of features as the data used to fit
        # TODO: Assert that clustering_model has predict method
        # TODO: Validate X
        n_samples, _ = X.shape
        labels = np.zeros(n_samples, dtype=np.uint32)
        queue = deque([(self.cluster_tree_, np.arange(n_samples))])
        while queue:
            node, indices = queue.popleft()
            if node.is_leaf:
                labels[indices] = node.label
            else:
                cluster = X[indices]
                clustering_model = node.clustering_model
                cluster_labels = clustering_model.predict(cluster)
                if hasattr(clustering_model, "n_clusters_"):
                    n_clusters = clustering_model.n_clusters_
                else:
                    n_clusters = len(np.unique(cluster_labels))
                for i in range(n_clusters):
                    child_indices = indices[np.nonzero(cluster_labels == i)[0]]
                    if child_indices.size > 0:
                        queue.append((node.children[i], child_indices))
        return labels
