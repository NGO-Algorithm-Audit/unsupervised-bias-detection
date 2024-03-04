import numpy as np
import heapq
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClusterMixin


class BiasAwareHierarchicalClustering(ABC, BaseEstimator, ClusterMixin):
    """
    Base class for Bias-Aware Hierarchical Clustering.

    This abstract class specifies an interface for all bias-aware hierarchical
    clustering classes.
    """

    def __init__(self, max_iter, min_cluster_size):
        self.max_iter = max_iter
        self.min_cluster_size = min_cluster_size

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
        n_samples, _ = X.shape
        self.n_clusters_ = 1
        labels = np.zeros(n_samples, dtype=np.uint32)
        clusters = []
        biases = []
        label = 0
        bias = -np.mean(y)
        heap = [(None, label, bias)]
        print(labels)
        for _ in range(self.max_iter):
            if not heap:
                break
            _, label, bias = heapq.heappop(heap)
            cluster_indices = np.nonzero(labels == label)[0]
            cluster = X[cluster_indices]
            cluster_labels = self._split(cluster)
            # TODO: Maybe check if cluster_labels are 0s and 1s
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
                    labels[indices1] = self.n_clusters_
                    self.n_clusters_ += 1
                else:
                    clusters.append(label)
                    biases.append(bias)
            else:
                clusters.append(label)
                biases.append(bias)
            print(labels)
            print(heap)
            print(clusters)
        clusters = np.array(clusters + [label for _, label, _ in heap])
        biases = np.array(biases + [bias for _, _, bias in heap])
        print(clusters)
        print(biases)
        indices = np.argsort(-biases)
        clusters = clusters[indices]
        self.biases_ = biases[indices]
        mapping = np.zeros(self.n_clusters_, dtype=np.uint32)
        mapping[clusters] = np.arange(self.n_clusters_, dtype=np.uint32)
        self.labels_ = mapping[labels]
        return self

    @abstractmethod
    def _split(self, X):
        """Splits the data into two clusters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        Returns
        -------
        labels : (n_samples)
            ndarray of shape (n_samples,)
        """
        pass
