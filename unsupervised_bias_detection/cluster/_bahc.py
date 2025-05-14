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
    """TODO: Add docstring.

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
        scores = []
        label = 0
        root = ClusterNode(label)
        self.cluster_tree_ = root
        # The entire dataset has a discrimination score of zero
        score = 0
        heap = [(None, root, score)]
        for _ in range(self.bahc_max_iter):
            if not heap:
                # If the heap is empty we stop iterating
                break
            # Take the cluster with the highest standard deviation of metric y
            _, node, score = heapq.heappop(heap)
            label = node.label
            cluster_indices = np.nonzero(labels == label)[0]
            cluster = X[cluster_indices]

            clustering_model = self.clustering_cls(**self.clustering_params)
            cluster_labels = clustering_model.fit_predict(cluster)

            # TODO: Generalize for more than 2 clusters
            # Can do this by checking clustering_model.n_clusters_ (if it exists)
            # or by checking the number of unique values in cluster_labels
            indices0 = cluster_indices[np.nonzero(cluster_labels == 0)[0]]
            indices1 = cluster_indices[np.nonzero(cluster_labels == 1)[0]]
            if (
                len(indices0) >= self.bahc_min_cluster_size
                and len(indices1) >= self.bahc_min_cluster_size
            ):
                # We calculate the discrimination scores using formula (1) in [1]
                # TODO: Move y[indices0] and y[indices1] into separate variables
                # to avoid recomputing them
                # Maybe create a function to compute the score
                mask0 = np.ones(n_samples, dtype=bool)
                mask0[indices0] = False
                score0 = np.mean(y[mask0]) - np.mean(y[indices0])
                mask1 = np.ones(n_samples, dtype=bool)
                mask1[indices1] = False
                score1 = np.mean(y[mask1]) - np.mean(y[indices1])
                if max(score0, score1) >= score:
                    std0 = np.std(y[indices0])
                    node0 = ClusterNode(label)
                    # heapq implements min-heap
                    # so we have to negate std before pushing
                    heapq.heappush(heap, (-std0, node0, score0))
                    std1 = np.std(y[indices1])
                    node1 = ClusterNode(self.n_clusters_)
                    heapq.heappush(heap, (-std1, node1, score1))
                    labels[indices1] = self.n_clusters_
                    # TODO: Increase n_clusters_ by clustering_model.n_clusters_ - 1
                    self.n_clusters_ += 1
                    children = [node0, node1]
                    node.split(clustering_model, children)
                else:
                    leaves.append(node)
                    scores.append(score)
            else:
                leaves.append(node)
                scores.append(score)
        if heap:
            # TODO: Check if this can be made more efficient
            leaves.extend((node for _, node, _ in heap))
            scores = np.concatenate([scores, [score for _, _, score in heap]])
        else:
            scores = np.array(scores)

        # We sort clusters by decreasing scores
        sorted_indices = np.argsort(-scores)
        self.scores_ = scores[sorted_indices]
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
