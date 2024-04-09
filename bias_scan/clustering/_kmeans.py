from ._bahc import BiasAwareHierarchicalClustering
from sklearn.cluster import KMeans


class BiasAwareHierarchicalKMeans(BiasAwareHierarchicalClustering):
    """Bias-Aware Hierarchical k-Means Clustering.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations.
    min_cluster_size : int
        Minimum size of a cluster.
    kmeans_params : dict
        k-means parameters
    """

    def __init__(
        self,
        max_iter,
        min_cluster_size,
        kmeans_params={"n_clusters": 2, "n_init": "auto"},
    ):
        super().__init__(max_iter, min_cluster_size)
        self.kmeans_params = kmeans_params
        self.kmeans = KMeans(**kmeans_params)

    def _split(self, X):
        return self.kmeans.fit_predict(X)
