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
    """

    def __init__(self, max_iter, min_cluster_size, kmodes_params={"n_clusters": 2}):
        super().__init__(max_iter, min_cluster_size)
        self.kmodes_params = kmodes_params
        self.kmodes = KModes(**kmodes_params)

    def _split(self, X):
        return self.kmodes.fit_predict(X)
