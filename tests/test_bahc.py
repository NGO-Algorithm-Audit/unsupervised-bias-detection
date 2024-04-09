import numpy as np

from bias_scan.clustering import BiasAwareHierarchicalKMeans


def test_shapes():
    # Checks that labels and biases have the right shapes
    rng = np.random.RandomState(12)
    X = rng.rand(20, 10)
    y = rng.rand(20)
    algo = BiasAwareHierarchicalKMeans(max_iter=5, min_cluster_size=2)
    algo.fit(X, y)
    assert len(algo.labels_) == 20
    assert len(algo.biases_) == algo.n_clusters_

def test_clusters():
    # Checks that label values are between 0 and n_clusters
    rng = np.random.RandomState(12)
    X = rng.rand(20, 10)
    y = rng.rand(20)
    algo = BiasAwareHierarchicalKMeans(max_iter=5, min_cluster_size=2)
    algo.fit(X, y)
    assert np.array_equal(np.unique(algo.labels_), np.arange(algo.n_clusters_))
