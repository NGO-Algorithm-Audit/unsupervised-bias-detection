from ._bahc import BiasAwareHierarchicalClustering
from kmodes.kmodes import KModes
from scipy import stats
import numpy as np

class BiasAwareHierarchicalKModes(BiasAwareHierarchicalClustering):
    """Bias-Aware Hierarchical k-Modes Clustering.

    Parameters
    ----------
    n_iter : int
        Number of iterations.
    min_cluster_size : int
        Minimum size of a cluster.
    kmodes_params : dict
        k-modes parameters
    
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
    >>> from unsupervised_bias_detection.clustering import BiasAwareHierarchicalKModes
    >>> import numpy as np
    >>> X = np.array([[0, 1], [0, 2], [0, 0], [1, 4], [1, 5], [1, 3]])
    >>> y = np.array([0, 0, 0, 10, 10, 10])
    >>> hbac = BiasAwareHierarchicalKModes(n_iter=1, min_cluster_size=1, random_state=12).fit(X, y)
    >>> hbac.labels_
    array([0, 0, 0, 1, 1, 1], dtype=uint32)
    >>> hbac.scores_
    array([ 10., -10.])
    """

    _dtype = None

    def __init__(self, n_iter, min_cluster_size, **kmodes_params):
        super().__init__(n_iter, min_cluster_size)

        if "n_clusters" in kmodes_params and kmodes_params["n_clusters"] != 2:
            raise ValueError(
                f"The parameter `n_clusters` should be 2, got {kmodes_params['n_clusters']}."
            )
        else:
            kmodes_params["n_clusters"] = 2
        
        print('kmodes_params is {}'.format(kmodes_params))

        self.kmodes = KModes(**kmodes_params)

    def _split(self, X):
        return self.kmodes.fit_predict(X)
    

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
            centroids[:, i] = stats.mode(X_label, axis=0)[0]

        return centroids
    
    def predict(self, X, seed=1):
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

            # calculate the hamming distance between the data and the centroid
            diff = (X != self.centroids_[:, k].T).astype(int)

            # Calculate the sum of squared differences
            distances[:, k] = np.sum(diff, axis=1)
        
        # in the cases that the minimum distance is not unique, a random label is assigned among the minimum distances
        min_distances = np.min(distances, axis=1)
        min_indices = np.where(distances == min_distances[:, None])
        labels = np.zeros(n_samples)

        # go through the samples
        for i in range(n_samples):
            # check if the minimum distance is unique
            clusters_with_min_distance_i = min_indices[1][min_indices[0] == i]

            # if unique, assign the label
            if len(clusters_with_min_distance_i) == 1:
                labels[i] =clusters_with_min_distance_i[0]
            else:
                # if not, assign a random label among the minimum distances
                np.random.seed(seed)
                labels[i] = np.random.choice(clusters_with_min_distance_i)
        
        
        return labels

    

    
