from ._bahc import BiasAwareHierarchicalClustering
from kmodes.kmodes import KModes


class BiasAwareHierarchicalKModes(BiasAwareHierarchicalClustering):
    """Bias-Aware Hierarchical k-Modes Clustering.

    Parameters
    ----------
    n_iter : int
        Number of iterations.
    min_cluster_size : int
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
    >>> hbac = BiasAwareHierarchicalKModes(n_iter=1, min_cluster_size=1, random_state=12).fit(X, y)
    >>> hbac.labels_
    array([0, 0, 0, 1, 1, 1], dtype=uint32)
    >>> hbac.scores_
    array([ 10., -10.])
    """

    _dtype = None

    def __init__(self, n_iter, min_cluster_size, **kmodes_params):
        super().__init__(n_iter, min_cluster_size)

        if "n_clusters" in kmodes_params and kmodes_params["n_clusters"] != 2:
            raise ValueError(
                f"The parameter `n_clusters` should be 2, got {kmodes_params['n_clusters']}."
            )
        else:
            kmodes_params["n_clusters"] = 2

        self.kmodes = KModes(**kmodes_params)

    def _split(self, X):
        return self.kmodes.fit_predict(X)
