import numpy as np

from unsupervised_bias_detection.clustering import BiasAwareHierarchicalKMeans


def test_shapes():
    # Checks that labels and biases have the right shapes
    rng = np.random.RandomState(12)
    X = rng.rand(20, 10)
    y = rng.rand(20)
    hbac = BiasAwareHierarchicalKMeans(n_iter=5, min_cluster_size=2)
    hbac.fit(X, y)
    assert len(hbac.labels_) == len(X)
    assert len(hbac.scores_) == hbac.n_clusters_

def test_labels():
    # Checks that label values are between 0 and n_clusters
    rng = np.random.RandomState(12)
    X = rng.rand(20, 10)
    y = rng.rand(20)
    hbac = BiasAwareHierarchicalKMeans(n_iter=5, min_cluster_size=2)
    hbac.fit(X, y)
    assert np.array_equal(np.unique(hbac.labels_), np.arange(hbac.n_clusters_))

def test_biases():
    # Checks that biases are sorted in descending order
    rng = np.random.RandomState(12)
    X = rng.rand(20, 10)
    y = rng.rand(20)
    hbac = BiasAwareHierarchicalKMeans(n_iter=5, min_cluster_size=2)
    hbac.fit(X, y)
    assert np.all(hbac.scores_[:-1] >= hbac.scores_[1:])