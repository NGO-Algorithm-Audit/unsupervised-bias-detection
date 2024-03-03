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
    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
    n_init : 'auto' or int, default='auto'
    kmeans_max_iter : int, default=300
    tol : float, default=1e-4
    """

    def __init__(
        self,
        max_iter,
        min_cluster_size,
        init="k-means++",
        n_init="auto",
        kmeans_max_iter=300,
        tol=1e-4,
    ):
        super().__init__(max_iter, min_cluster_size)
        self.init = init
        self.n_init = n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.tol = tol

    def split(self, X):
        kmeans = KMeans(
            n_clusters=2,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.kmeans_max_iter,
            tol=self.tol,
        )
        return kmeans.fit_predict(X)
