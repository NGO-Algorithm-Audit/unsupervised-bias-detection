from ._bahc import BiasAwareHierarchicalClustering
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans


class BiasAwareHierarchicalKMeans(BaseEstimator, ClusterMixin):
    """Bias-Aware Hierarchical k-Means Clustering.

    Parameters
    ----------
    bahc_max_iter : int
        Maximum number of iterations.
    bahc_min_cluster_size : int
        Minimum size of a cluster.
    kmeans_params : dict
        k-means parameters

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm.
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point. Lower labels correspond to higher discrimination scores.
    scores_ : ndarray of shape (n_clusters_,)
        Discrimination scores for each cluster.

    References
    ----------
    .. [1] J. Misztal-Radecka, B. Indurkhya, "Bias-Aware Hierarchical Clustering for detecting the discriminated
           groups of users in recommendation systems", Information Processing & Management, vol. 58, no. 3, May. 2021.

    Examples
    --------
    >>> from unsupervised_bias_detection.clustering import BiasAwareHierarchicalKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> y = np.array([0, 0, 0, 10, 10, 10])
    >>> bahc = BiasAwareHierarchicalKMeans(bahc_max_iter=1, bahc_min_cluster_size=1, random_state=12).fit(X, y)
    >>> bahc.labels_
    array([0, 0, 0, 1, 1, 1], dtype=uint32)
    >>> bahc.scores_
    array([ 10., -10.])
    """

    def __init__(
        self,
        bahc_max_iter,
        bahc_min_cluster_size,
        **kmeans_params,
    ):
        # TODO: Remove this once we have a better way to handle the number of clusters
        if "n_clusters" in kmeans_params and kmeans_params["n_clusters"] != 2:
            raise ValueError(
                f"The parameter `n_clusters` should be 2, got {kmeans_params['n_clusters']}."
            )
        else:
            kmeans_params["n_clusters"] = 2

        if "n_init" not in kmeans_params:
            kmeans_params["n_init"] = "auto"

        self.bahc_max_iter = bahc_max_iter
        self.bahc_min_cluster_size = bahc_min_cluster_size
        self._bahc = BiasAwareHierarchicalClustering(
            KMeans,
            bahc_max_iter,
            bahc_min_cluster_size,
            **kmeans_params,
        )

    def fit(self, X, y):
        self._bahc.fit(X, y)
        self.n_clusters_ = self._bahc.n_clusters_
        self.labels_ = self._bahc.labels_
        self.scores_ = self._bahc.scores_
        self.cluster_tree_ = self._bahc.cluster_tree_
        return self
    
    def predict(self, X):
        return self._bahc.predict(X)