import numpy as np
import heapq
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClusterMixin


class BiasAwareHierarchicalClustering(ABC, BaseEstimator, ClusterMixin):
    """
    Base class for Bias Aware Hierarchical Clustering.

    This abstract class specifies an interface for all bias aware hierarchical
    clustering classes.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations.
    min_cluster_size : int
        Minimum size of a cluster.
    """

    def __init__(self, max_iter, min_cluster_size):
        self.max_iter = max_iter
        self.min_cluster_size = min_cluster_size

    def fit(self, X, y):
        """What the function does

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
            Description here
        """
        n_samples, _ = X.shape
        self.n_clusters_ = 1
        self.labels_ = np.zeros(n_samples, dtype=np.uint16)
        labels = []
        biases = []
        label = 0
        bias = -np.mean(y)
        heap = [(None, label, bias)]
        for _ in range(self.max_iter):
            if not heap:
                break
            _, label, bias = heapq.heappop(heap)
            cluster_indices = np.nonzero(self.labels_ == label)[0]
            cluster = X[cluster_indices]
            cluster_labels = self.split(cluster)
            indices0 = cluster_indices[np.nonzero(cluster_labels == 0)[0]]
            indices1 = cluster_indices[np.nonzero(cluster_labels == 1)[0]]
            if (
                len(indices0) >= self.min_cluster_size
                and len(indices1) >= self.min_cluster_size
            ):
                mask0 = np.ones(n_samples, dtype=bool)
                mask0[indices0] = False
                bias0 = np.mean(y[mask0]) - np.mean(y[indices0])
                mask1 = np.ones(n_samples, dtype=bool)
                mask1[indices1] = False
                bias1 = np.mean(y[mask1]) - np.mean(y[indices1])
                if max(bias0, bias1) >= bias:
                    std0 = np.std(y[indices0])
                    heapq.heappush(heap, (-std0, label, bias0))
                    std1 = np.std(y[indices1])
                    heapq.heappush(heap, (-std1, self.n_clusters_, bias1))
                    self.labels_[indices1] = self.n_clusters_
                    self.n_clusters_ += 1
                else:
                    labels.append(label)
                    biases.append(bias)
            else:
                labels.append(label)
                biases.append(bias)
        labels = np.array(labels + [label for _, label, _ in heap])
        biases = np.array(biases + [bias for _, _, bias in heap])
        self.biases_ = biases[np.argsort(labels)]
        return self

    @abstractmethod
    def split(self, X):
        """What the function does

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        Returns
        -------
        labels : (n_samples)
            Description goes here.
        """
        pass
