import numpy as np
import heapq
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.stats import ttest_ind
from scipy import stats
from sklearn.metrics import calinski_harabasz_score as CH_score
from mpmath import mp

class BiasAwareHierarchicalClustering(ABC, BaseEstimator, ClusterMixin):
    """
    Base class for Bias-Aware Hierarchical Clustering.

    This abstract class specifies an interface for all bias-aware hierarchical clustering classes.

    References
    ----------
    .. [1] J. Misztal-Radecka, B. Indurkhya, "Bias-Aware Hierarchical Clustering for detecting the discriminated
           groups of users in recommendation systems", Information Processing & Management, vol. 58, no. 3, May. 2021.
    """

    def __init__(self, n_iter, min_cluster_size):
        self.n_iter = n_iter
        self.min_cluster_size = min_cluster_size

    
    def fit(self, X, y, random_state=None):
        """Compute bias-aware hierarchical clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : array-like of shape (n_samples)
            Metric values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = self._validate_data(
            X, y, reset=False, accept_large_sparse=False, dtype=self._dtype, order="C"
        )
        n_samples, _ = X.shape
        # We start with all samples in a single cluster
        self.n_clusters_ = 1
        # We assign all samples a label of zero
        labels = np.zeros(n_samples, dtype=np.uint32)
        clusters = []
        scores = []
        label = 0
        # The entire dataset has a discrimination score of zero
        score = 0
        heap = [(None, label, score)]
        for _ in range(self.n_iter):
            if not heap:
                # If the heap is empty we stop iterating
                break
            # Take the cluster with the highest standard deviation of metric y
            _, label, score = heapq.heappop(heap)
            cluster_indices = np.nonzero(labels == label)[0]
            cluster = X[cluster_indices]
            cluster_labels = self._split(cluster)
            indices0 = cluster_indices[np.nonzero(cluster_labels == 0)[0]]
            indices1 = cluster_indices[np.nonzero(cluster_labels == 1)[0]]
            if (
                len(indices0) >= self.min_cluster_size
                and len(indices1) >= self.min_cluster_size
            ):
                # We calculate the discrimination scores using formula (1) in [1]
                mask0 = np.ones(n_samples, dtype=bool)
                mask0[indices0] = False
                score0 = np.mean(y[mask0]) - np.mean(y[indices0])
                mask1 = np.ones(n_samples, dtype=bool)
                mask1[indices1] = False
                score1 = np.mean(y[mask1]) - np.mean(y[indices1])
                if max(score0, score1) >= score:
                    # heapq implements min-heap
                    # so we have to negate std before pushing
                    std0 = np.std(y[indices0])
                    heapq.heappush(heap, (-std0, label, score0))
                    std1 = np.std(y[indices1])
                    heapq.heappush(heap, (-std1, self.n_clusters_, score1))
                    labels[indices1] = self.n_clusters_
                    self.n_clusters_ += 1
                else:
                    clusters.append(label)
                    scores.append(score)
            else:
                clusters.append(label)
                scores.append(score)
        clusters = np.array(clusters + [label for _, label, _ in heap])
        scores = np.array(scores + [score for _, _, score in heap])
        # We sort clusters by decreasing scores
        indices = np.argsort(-scores)
        clusters = clusters[indices]
        self.scores_ = scores[indices]
        mapping = np.zeros(self.n_clusters_, dtype=np.uint32)
        mapping[clusters] = np.arange(self.n_clusters_, dtype=np.uint32)
        self.labels_ = mapping[labels]


        # Fit the centroids
        self.centroids_ = self.calc_centroids(X, self.labels_)

        return self


    @abstractmethod
    def _split(self, X, random_state=None):
        """Split the data into two clusters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each point. Every label is either 0 or 1 indicating
            that the point belongs to the first or the second cluster, respectively.
        """
        pass

    def binary_chi_square_test(self, m, labels, k, bonf_correct):
        """
        Performs a chi-square test of independence for two binary samples.
        
        Parameters:
        -----------
        m: np.array
            The metric values
        labels: np.array
            The cluster labels
        k: int
            The cluster number
        bonf_correct: bool
            Whether to apply Bonferroni correction
            
        Returns:
        --------
            - p-value
            - difference in proportions
            - proportion of 1s in cluster 1
            - proportion of 1s in cluster 0
        """

        # take the idx where labels == k
        idx = (labels == k)

        # calculate the n_clust
        n_clust = len(np.unique(labels))

        # get the target values for the two clusters
        m1, m0 = m[idx], m[~idx]

        # Input validation
        if not all(np.isin(m, [0, 1])):
            raise ValueError("All values must be binary (0 or 1)")
            
        # Create contingency table
        a = np.sum(m1 == 1)  # count of 1s in y
        b = np.sum(m0 == 1)  # count of 1s in z
        c = np.sum(m1 == 0)  # count of 0s in y
        d = np.sum(m0 == 0)  # count of 0s in z
        
        contingency_table = np.array([[a, b],
                                    [c, d]])
        
        # Perform chi-square test
        test_statistic, p_clust, dof = stats.chi2_contingency(contingency_table)[:3]
       

        if bonf_correct and n_clust > 2:
            p_clust = p_clust * n_clust
        
        # calc difference in proportions
        p1 = a / (a + c)
        p0 = b / (b + d)
        diff_clust = p1 - p0
        
        return p_clust, diff_clust, p1, p0


    def t_test(self, m, labels, k, bonf_correct, alternative='two-sided'):

        # take the idx where labels == k
        idx = (labels == k)

        # calculate the n_clust
        n_clust = len(np.unique(labels))

        # get the target values for the two clusters
        m1, m0 = m[idx], m[~idx]
    
        # calculate the p-value using a t-test
        t, p_clust = ttest_ind(m1, m0, equal_var=False, alternative=alternative)

        # if the p-value is 0, we have numerical underflow
        if p_clust == 0:

            t = mp.mpf(t)
            nu = mp.mpf(res.df)
            x2 = nu / (t**2 + nu)
            p_clust = mp.betainc(nu/2, mp.one/2, x2=x2, regularized=True)
            print('original p is zero, accounting for underflow')

        # apply bonferroni correction
        if bonf_correct and n_clust > 2: # if there are two clusters, just use the single p-value
            p_clust = p_clust * n_clust # Bonferroni correction - multiply by number of clusters

        # calculate the difference in means
        diff_clust = m1.mean() - m0.mean() # Difference in means
        
        return p_clust, diff_clust, m1.mean(), m0.mean()
    

    def calc_ratio_within_between(self, m, labels):

        # loop over each cluster, record within cluster var, average, and number of samples
        within = []
        averages = []
        n_k = []


        # calculate the total number of samples and the number of clusters
        n = len(m)
        K = len(np.unique(labels))
        overall_mean = np.mean(m)

        # loop over each cluster
        for k in np.unique(labels):

            # take the idx where labels == k
            idx = (labels == k)

            # get m when for cluster k, calculate the mean
            m_k = m[idx]
            mean_k = np.mean(m_k)

            # calculate the within cluster variance
            within_k  = np.sum((m_k - mean_k)**2)

            # append to the list
            within.append(within_k)

            # append the average and the number of samples
            averages.append(mean_k)
            n_k.append(len(m_k))
        
        # calculate the between cluster variance
        B = np.sum([n_k[i]*(averages[i] - overall_mean)**2 for i in range(len(averages))])
        B_adj = B / (K - 1) # divide between degrees of freedom

        # calculate the within cluster variance
        W = np.sum(within)
        W_adj = W / (n - K) # divide within degrees of freedom

        # calculate the ratio
        ratio = B_adj / W_adj

        return ratio, B, W

        


        



            
