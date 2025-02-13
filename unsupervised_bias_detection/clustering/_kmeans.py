import numpy as np
from ._bahc import BiasAwareHierarchicalClustering
from sklearn.cluster import KMeans


class BiasAwareHierarchicalKMeans(BiasAwareHierarchicalClustering):
    """Bias-Aware Hierarchical k-Means Clustering.

    Parameters
    ----------
    n_iter : int
        Number of iterations.
    min_cluster_size : int
        Minimum size of a cluster.
    kmeans_params : dict
        k-means parameters
    
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
    >>> from unsupervised_bias_detection.clustering import BiasAwareHierarchicalKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> y = np.array([0, 0, 0, 10, 10, 10])
    >>> hbac = BiasAwareHierarchicalKMeans(n_iter=1, min_cluster_size=1, random_state=12).fit(X, y)
    >>> hbac.labels_
    array([0, 0, 0, 1, 1, 1], dtype=uint32)
    >>> hbac.scores_
    array([ 10., -10.])
    """

    _dtype = [np.float32, np.float64]

    def __init__(
        self,
        n_iter,
        min_cluster_size,
        **kmeans_params,
    ):
        super().__init__(n_iter, min_cluster_size)

        if "n_clusters" in kmeans_params and kmeans_params["n_clusters"] != 2:
            raise ValueError(
                f"The parameter `n_clusters` should be 2, got {kmeans_params['n_clusters']}."
            )
        else:
            kmeans_params["n_clusters"] = 2
        
        if "n_init" not in kmeans_params:
            kmeans_params["n_init"] = "auto"
        
        self.kmeans = KMeans(**kmeans_params)

    def _split(self, X):
        return self.kmeans.fit_predict(X)
    

    def calc_centroids(self, X, labels):
        """ Calculate the centroids of the clusters based on the labels

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        labels : array-like of shape (n_samples)
            Cluster labels for each point.
        
        """

        # create an array of (d, k) with d being the number of features and k the number of unique labels
        centroids = np.zeros((X.shape[1], len(np.unique(labels))))

        # iterate over the labels
        for i, label in enumerate(np.unique(labels)):

            # get the data points that belong to the cluster with the current label
            X_label = X[labels == label]

            # calculate the mean of the data points
            centroids[:, i] = np.mean(X_label, axis=0)

        return centroids
    
    def predict(self, X):
        """Predict the cluster labels for the samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array-like of shape (n_samples,)
            Cluster labels for each point.
        """

        # Validate the data
        X = self._validate_data(
            X, reset=False, accept_large_sparse=False, dtype=self._dtype, order="C"
        )
        # Get dimensions
        n_samples = X.shape[0]
        n_clusters = self.centroids_.shape[1]
        
        # Initialize distance matrix
        distances = np.zeros((n_samples, n_clusters))
        
        # Compute squared Euclidean distance between each sample and each centroid
        for k in range(n_clusters):

            # get the centroid
            centroid_k = self.centroids_[:, k].T

            # Deduct the centroid from the data
            diff = X - centroid_k

            # Calculate the sum of squared differences
            distances[:, k] = np.sum(diff * diff, axis=1)
        
        # Get the index (label) of the closest centroid for each sample
        labels = np.argmin(distances, axis=1)
        
        
        return labels

    

    