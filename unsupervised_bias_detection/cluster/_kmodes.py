from ._bahc import BiasAwareHierarchicalClustering
from kmodes.kmodes import KModes
from sklearn.base import BaseEstimator, ClusterMixin


class BiasAwareHierarchicalKModes(BaseEstimator, ClusterMixin):
    """Bias-Aware Hierarchical k-Modes Clustering.

    Parameters
    ----------
    bahc_max_iter : int
        Maximum number of iterations.
    bahc_min_cluster_size : int
        Minimum size of a cluster.
    kmodes_params : dict
        k-modes parameters

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
    >>> from unsupervised_bias_detection.clustering import BiasAwareHierarchicalKModes
    >>> import numpy as np
    >>> X = np.array([[0, 1], [0, 2], [0, 0], [1, 4], [1, 5], [1, 3]])
    >>> y = np.array([0, 0, 0, 10, 10, 10])
    >>> bahc = BiasAwareHierarchicalKModes(bahc_max_iter=1, bahc_min_cluster_size=1, random_state=12).fit(X, y)
    >>> bahc.labels_
    array([0, 0, 0, 1, 1, 1], dtype=uint32)
    >>> bahc.scores_
    array([ 10., -10.])
    """

    def __init__(self, bahc_max_iter, bahc_min_cluster_size, **kmodes_params):
        # TODO: Remove this once we have a better way to handle the number of clusters
        if "n_clusters" in kmodes_params and kmodes_params["n_clusters"] != 2:
            raise ValueError(
                f"The parameter `n_clusters` should be 2, got {kmodes_params['n_clusters']}."
            )
        else:
            kmodes_params["n_clusters"] = 2

        self.bahc_max_iter = bahc_max_iter
        self.bahc_min_cluster_size = bahc_min_cluster_size
        self._hbac = BiasAwareHierarchicalClustering(
            KModes, bahc_max_iter, bahc_min_cluster_size, **kmodes_params
        )

    def fit(self, X, y):
        self._hbac.fit(X, y)
        self.n_clusters_ = self._hbac.n_clusters_
        self.labels_ = self._hbac.labels_
        self.scores_ = self._hbac.scores_
        self.cluster_tree_ = self._hbac.cluster_tree_
        return self

    def predict(self, X):
        return self._hbac.predict(X)