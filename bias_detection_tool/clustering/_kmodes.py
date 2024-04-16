from ._bahc import BiasAwareHierarchicalClustering
from kmodes.kmodes import KModes


class BiasAwareHierarchicalKModes(BiasAwareHierarchicalClustering):
    """Bias-Aware Hierarchical k-Modes Clustering.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations.
    min_cluster_size : int
        Minimum size of a cluster.
    kmodes_params : dict
        k-modes parameters
    
    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm.
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point.
    biases_ : ndarray of shape (n_clusters_,)
        Bias values for each cluster.
    
    References
    ----------
    .. [1] J. Misztal-Radecka, B. Indurkhya, "Bias-Aware Hierarchical Clustering for detecting the discriminated
           groups of users in recommendation systems", Information Processing & Management, vol. 58, no. 3, May. 2021.
    
    Examples
    --------
    >>> from bias_detection_tool.clustering import BiasAwareHierarchicalKModes
    >>> import numpy as np
    >>> X = np.array([[0, 1], [0, 2], [0, 0], [1, 4], [1, 5], [1, 3]])
    >>> y = np.array([0, 0, 0, 10, 10, 10])
    >>> bias_aware_kmodes = BiasAwareHierarchicalKModes(max_iter=1, min_cluster_size=1, random_state=12).fit(X, y)
    >>> bias_aware_kmodes.labels_
    array([0, 0, 0, 1, 1, 1], dtype=uint32)
    >>> bias_aware_kmodes.biases_
    array([ 10., -10.])
    """

    def __init__(self, max_iter, min_cluster_size, **kmodes_params):
        super().__init__(max_iter, min_cluster_size)

        if "n_clusters" in kmodes_params and kmodes_params["n_clusters"] != 2:
            raise ValueError(
                f"The parameter `n_clusters` should be 2, got {kmodes_params['n_clusters']}."
            )
        else:
            kmodes_params["n_clusters"] = 2

        self.kmodes = KModes(**kmodes_params)

    def _split(self, X):
        return self.kmodes.fit_predict(X)
