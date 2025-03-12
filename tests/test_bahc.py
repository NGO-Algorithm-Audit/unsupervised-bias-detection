import numpy as np
from unsupervised_bias_detection.cluster import BiasAwareHierarchicalKMeans


def test_shapes():
    # Checks that labels and biases have the right shapes
    rng = np.random.RandomState(12)
    X = rng.rand(20, 10)
    y = rng.rand(20)
    bahc = BiasAwareHierarchicalKMeans(bahc_max_iter=5, bahc_min_cluster_size=2)
    bahc.fit(X, y)
    assert len(bahc.labels_) == len(X)
    assert len(bahc.scores_) == bahc.n_clusters_


def test_labels():
    # Checks that label values are between 0 and n_clusters
    rng = np.random.RandomState(12)
    X = rng.rand(20, 10)
    y = rng.rand(20)
    bahc = BiasAwareHierarchicalKMeans(bahc_max_iter=5, bahc_min_cluster_size=2)
    bahc.fit(X, y)
    assert np.array_equal(np.unique(bahc.labels_), np.arange(bahc.n_clusters_))


def test_biases():
    # Checks that biases are sorted in descending order
    rng = np.random.RandomState(12)
    X = rng.rand(20, 10)
    y = rng.rand(20)
    bahc = BiasAwareHierarchicalKMeans(bahc_max_iter=5, bahc_min_cluster_size=2)
    bahc.fit(X, y)
    assert np.all(bahc.scores_[:-1] >= bahc.scores_[1:])
