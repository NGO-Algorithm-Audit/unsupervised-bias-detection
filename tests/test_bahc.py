import numpy as np
from unsupervised_bias_detection.cluster import BiasAwareHierarchicalKMeans


def test_shapes():
    # Checks that labels and scores have the right shapes
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


# def test_cluster_sizes():
    # Checks that cluster sizes are at least bahc_min_cluster_size


def test_constant_metric():
    # Checks that there is only one cluster with a score of 0 if the metric is constant
    rng = np.random.RandomState(12)
    X = rng.rand(20, 10)
    y = np.full(20, rng.rand())
    bahc = BiasAwareHierarchicalKMeans(bahc_max_iter=5, bahc_min_cluster_size=2)
    bahc.fit(X, y)
    assert bahc.n_clusters_ == 1
    assert bahc.scores_[0] == 0


def test_scores():
    # Checks that scores are computed correctly
    rng = np.random.RandomState(12)
    X = rng.rand(20, 10)
    y = rng.rand(20)
    bahc = BiasAwareHierarchicalKMeans(bahc_max_iter=5, bahc_min_cluster_size=2)
    bahc.fit(X, y)
    # TODO: Check this!!!
    for i in range(bahc.n_clusters_):
        cluster_indices = np.arange(20)[bahc.labels_ == i]
        complement_indices = np.arange(20)[bahc.labels_ != i]
        score = np.mean(y[complement_indices]) - np.mean(y[cluster_indices])
        assert bahc.scores_[i] == score


def test_scores_are_sorted():
    # Checks that scores are sorted in descending order
    rng = np.random.RandomState(12)
    X = rng.rand(20, 10)
    y = rng.rand(20)
    bahc = BiasAwareHierarchicalKMeans(bahc_max_iter=5, bahc_min_cluster_size=2)
    bahc.fit(X, y)
    assert np.all(bahc.scores_[:-1] >= bahc.scores_[1:])


def test_predict():
    # Checks that predict returns the same labels as fit
    rng = np.random.RandomState(12)
    X = rng.rand(20, 10)
    y = rng.rand(20)
    bahc = BiasAwareHierarchicalKMeans(bahc_max_iter=5, bahc_min_cluster_size=2)
    bahc.fit(X, y)
    assert np.array_equal(bahc.predict(X), bahc.labels_)
