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
    init : {'Huang', 'Cao', 'random'}, default='Cao'
    n_init : int, default=10
    kmodes_max_iter : int, default=100
    """

    def __init__(
        self,
        max_iter,
        min_cluster_size,
        init="Cao",
        n_init=10,
        kmodes_max_iter=100,
    ):
        super().__init__(max_iter, min_cluster_size)
        self.init = init
        self.n_init = n_init
        self.kmodes_max_iter = kmodes_max_iter

    def split(self, X):
        kmodes = KModes(
            n_clusters=2,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.kmodes_max_iter,
        )
        return kmodes.fit_predict(X)
