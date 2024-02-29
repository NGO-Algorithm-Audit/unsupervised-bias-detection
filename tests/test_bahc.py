import numpy as np

from bias_scan.clustering import BiasAwareHierarchicalKMeans


def test_clusters():
    # Checks that label values are between 0 and n_clusters
    rng = np.random.RandomState(12)
    X = rng.rand(10, 5)
    y = rng.rand(10)
    algo = BiasAwareHierarchicalKMeans(max_iter=3, min_cluster_size=2)
    algo.fit(X, y)
    assert np.array_equal(np.unique(algo.labels_), np.arange(algo.n_clusters_))
