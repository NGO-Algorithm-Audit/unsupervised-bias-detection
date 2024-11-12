from copy import copy
from itertools import product
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings("error")

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_ind
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from unsupervised_bias_detection.clustering import BiasAwareHierarchicalKMeans
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from joblib import Parallel, delayed
import sys

def set_seed(seed):
    np.random.seed(seed)


METRICS_BINARY = {
    'y_pred': lambda y, y_pred: y_pred,
    'fp': lambda y, y_pred: np.logical_and(y == 0, y_pred == 1).astype(int),
    'fn': lambda y, y_pred: np.logical_and(y == 1, y_pred == 0).astype(int),
    'err': lambda y, y_pred: (y != y_pred).astype(int)
}

METRICS_CONTINUOUS = {
    'y_pred': lambda y, y_pred: y_pred,
    'err': lambda y, y_pred: (y - y_pred)**2
}


def simulate_synthetic_data( K, N, y_dgp, x_dgp, d, seed, binary_y=False):

    """
    Follows the DGP from Misztal-Radecka & Indurkhya

    Arguments:
        - K: either 
            int: number of segments
            str: 'random', then uniformly sample K from [2  10]
        - N: either
            int: number of observations
            str: 'random', then uniformly sample per segment N_k from [10  200]
        - y_dgp: str, one of the following options
            random: M \sim N(mu_k, sigma_k), where mu_k, sigma_k are sampled from [0  1]
            constant: M \sim N(0, 1) for all segments
            linear: M \sim N(0, 1) + 0.1 * k, e.g. an increasing trend per segment
        - x_dgp: str, one of the following options
            constant: X \sim N(0, 1) for all segments
        - d: int, number of features
            Standard is 2 from Misztal-Radecka & Indurkhya
        - seed: int, seed for reproducibility
        - binary_y: bool, whether to make the target variable binary

    """

    # before starting, set the seed
    set_seed(seed)

    
    # if K is random, sample uniformly from [2  10]
    if K == 'random':
        K = np.random.randint(2, 10)
    
    # if N_k is random, sample uniformly from [10  200] for each segment
    if N == 'random':
        N_k_values = np.random.randint(10, 200, K)
    else:
        N_k_values = [int(N/K)]*K
    
    # create matrix to store the data with shape (N, d+1, K)
    data = np.zeros((np.sum(N_k_values), d+2))
    i = 0

    # loop over each segment
    for k in range(K):
        
        # get the number of observations for this segment
        n_k = N_k_values[k]

        # update the index
        i += n_k

        ## Set the dgp for the target variable y - this is based on whether or not the target variable is binary
        # For a continuous target variable, we sample mu_k, sigma_k from [0  1] and sample from N(mu_k, sigma_k)
        if not binary_y:
            # If random, sample mu_k, sigma_k from [0  1], and sample from N(mu_k, sigma_k)
            if y_dgp == 'random':
                mu_k, sigma_k = np.random.uniform(0, 1, 2)
        
            # If constant, set mu_k as 0, sigma_k as 1, and sample from N(0, 1)
            elif y_dgp == 'constant':
                mu_k, sigma_k = 0, 1
            
            # If linear, set mu_k between -1 and 1, and increase linearly with k
            elif y_dgp == 'linear':

                # set sigma_k as 1
                sigma_k = 1

                # increase linearly with k
                mu_k = -1 + (2 * k / (K-1))
                

            # now generate the target variable y from N(mu_k, sigma_k)
            y_k = np.random.normal(mu_k, sigma_k, n_k)
        else:
            
            # If binary, and random, sample mu_k from [0  1], and set the probability of y=1 as mu_k
            if y_dgp == 'random':
                p_k = np.random.uniform(0, 1, 1)

            # If binary, and constant, set mu_k as 0.5
            elif y_dgp == 'constant':
                p_k = 0.5

            # If binary, and linear, set mu_k between 0.1 and 0.9, and increase linearly with k
            elif y_dgp == 'linear':
                p_k = 0.1 + 0.8 * k / (K-1)
            
            # generate the binary target variable y from a binomial distribution with probability p_k
            y_k = np.random.binomial(1, p_k, n_k)        

        # if binary_y is True, then make the target variable binary
        if binary_y:
            y_k = np.where(y_k > 0.5, 1, 0)

        # then, define the DGP for the features X
        if x_dgp == 'constant':
            mu_k_x = np.zeros(d)
            sigma_k_x = np.eye(d)
        elif x_dgp == 'random':

            # sample mu from [0  1]
            mu = np.random.uniform(0, 1, 1)
            mu_k_x = np.ones(d) * mu

            # set the covariance matrix as the identity
            sigma_k_x = np.eye(d) 

        
        # generate the features as a matrix X
        X_k = np.random.multivariate_normal(mu_k_x, sigma_k_x, n_k)

        # create an entry to the matrix
        entry = np.zeros((n_k, d+2))
        entry[:, 0] = y_k # first column is the target variable
        entry[:, 1:-1] = X_k # up until the last column are the features
        entry[:, -1] = k # last column is the segment number

        # save the entry in the data matrix
        data[i-n_k:i, :] = entry


    # save the information in a dictionary
    sim_dict = {
        'data': data,
        'K': K,
        'N_k': N_k_values,
        'y_dgp': y_dgp,
        'x_dgp': x_dgp,
        'd': d
    }

    return sim_dict

def simulate_hbac(method,  K, N, y_dgp, x_dgp,  d, seed, binary_y=False, fit_train=True, target_col='y_pred', n_iter_hbac='known_clusters', min_cluster_size=5, val_frac=0.5):
    """
    Simulates the synthetic data, fits the clustering method, and returns the results

    Arguments:
        - method: str, one of the following options
            kmeans: KMeans clustering
            hbac: BiasAwareHierarchicalKMeans
            randomclusters: Randomly assign cluster labels
        - K: int, 
            number of segments
        - N: int, 
            number of observations
        - y_dgp: str, 
            one of the following options
            random: M \sim N(mu_k, sigma_k), where mu_k, sigma_k are sampled from [0  1]
            constant: M \sim N(0, 1) for all segments
            linear: M \sim N(0, 1) + 0.1 * k, e.g. an increasing trend per segment
        - x_dgp: str,
            One of the following options
            constant: X \sim N(0, 1) for all segments
            random: X \sim N(\mu_k, 1) for all segments, with \mu_k sampled from [0  1]
     
        - d: int, 
            number of features
        - seed: int,
            seed for reproducibility
        - binary_y: bool, 
            whether to make the target variable binary
        - fit_train: bool,
            whether to fit the model on the training set
        - target_col: str,
            optional, one of the following options
            y_pred: predicted target variable
            err: error
            fp: false positives, only for binary target variables
            fn: false negatives, only for binary target variables
        - n_iter_hbac: int or str,
            number of iterations for the BiasAwareHierarchicalKMeans, or 'known_clusters' to use the number of known clusters
        - min_cluster_size: int,
            minimum cluster size for the BiasAwareHierarchicalKMeans
        - val_frac: float,
            fraction of data to use for validation

    """

    # if fit_train is true, multiply the total data by the validation fraction
    if fit_train and N != 'random':
        N = int(N / val_frac)

    # simulate the synthetic data
    result_sim = simulate_synthetic_data(K, N, y_dgp, x_dgp, d, seed, binary_y=binary_y)
    X = result_sim['data'][:, 1:-1]
    y = result_sim['data'][:, 0].flatten()
    k = result_sim['data'][: , -1].flatten()

    # Split in train and val set
    if fit_train:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_frac, stratify=k, random_state=seed)
    else:
        X_train, X_val, y_train, y_val = X, X, y, y # in this case, we use all the data for training/testing

    # If target_col == y, no need to fit the model on the training set
    if target_col == 'y':

        # the target is equivalent to the y
        m_val = y_val
        m_train = y_train

    else:

        # define the model
        if binary_y:
            model = make_pipeline(StandardScaler(), LogisticRegression())
        else:
            model = make_pipeline(StandardScaler(), LinearRegression())
        
        # Fit the model on the training set
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val) # predicted labels on the validation set
        y_pred_train = model.predict(X_train) # predicted labels on the training set
    
        # if binary_y is True, then define target via binary metrics
        if binary_y:
            m_val = METRICS_BINARY[target_col](y_val, y_pred_val)
            m_train = METRICS_BINARY[target_col](y_train, y_pred_train)
        # Otherwise, define target via continuous metrics
        else:
            m_val = METRICS_CONTINUOUS[target_col](y_val, y_pred_val)
            m_train = METRICS_CONTINUOUS[target_col](y_train, y_pred_train)
        
    # obtain cluster labels
    if method in ['kmeans', 'kmeans_cv']:

        # define the cluster model
        set_seed(seed)

        # if n_iter_hbac is 'known_clusters', use the number of known clusters
        if n_iter_hbac == 'known_clusters':
            n_iter_kmeans = result_sim['K']
        else:
            n_iter_kmeans = K
        
        cluster_model = KMeans(n_clusters=n_iter_kmeans)
        cluster_model.fit(X_train)

        # fit the model on the training set
        if method == 'kmeans':
            labels = cluster_model.predict(X_val)
        else:
            labels = cluster_model.fit_predict(X_val)
    
    # if method is hbac, use the BiasAwareHierarchicalKMeans
    elif method == 'hbac':
        set_seed(seed)

        # if n_iter_hbac is 'known_clusters', use the number of known clusters
        if n_iter_hbac == 'known_clusters':
            n_iter_hbac = result_sim['K']-1
        
        hbac = BiasAwareHierarchicalKMeans(n_iter=n_iter_hbac, min_cluster_size=min_cluster_size) # 5 is the minimum

        # Fit on training set
        hbac.fit(X_train, m_train)

        # Fit on training set, predict on validation set
        if fit_train:
            labels = hbac.predict(X_val)
        # Else, fit/predict on the same set
        else:
            labels = hbac.labels_

    # if method is randomclusters, randomly assign cluster labels
    elif method == 'randomclusters':
        K_for_random = result_sim['K']
        labels = np.random.choice(range(K_for_random), size=X_val.shape[0])
    else:
        raise ValueError(f"Not a known method ({method})")
    

    return X_val, X_train, m_val, m_train, labels, y_val, y_train


def check_before_test(c0, c1, min_samples=5):
    """Determines whether to do significance testing
    (don't do this with too few observations/constant data to avoid NaNs)."""
    if (c0.size < min_samples) or (c1.size < min_samples):
        return True

    if np.mean(c0) == np.mean(c1):
        return True
    
    if np.all(c0 == c0[0]) or np.all(c1 == c1[0]):
        return True

    return False



def null_X_y(X_train, X_val, y_train, y_val, n_perm=1000, perm=True):
    """
    Return n_perm pairs of (X, y_perm) where y_perm is a permutation of y

    Args:
        X_train: np.array, shape (N, d)
        X_val: np.array, shape (N, d)
        y_train: np.array, shape (N,)
        y_val: np.array, shape (N,)
    """
    null_X_y = []
    for i in range(n_perm):

        # permute the y values if perm is True
        if perm:
            y_train_perm = np.random.permutation(y_train)
            y_val_perm = np.random.permutation(y_val)
        else:
            y_train_perm = y_train
            y_val_perm = y_val

        # create the pair
        pairs = (X_train, X_val, y_train_perm, y_val_perm)
        null_X_y.append(pairs)

    return null_X_y

def fit_hbac(X_train, y_train, n_iter_hbac, min_cluster_size, seed):
    """
    Fit the HBAC model on the training set

    Args:
        X_train: np.array, shape (N, d)
        y_train: np.array, shape (N,)
        n_iter_hbac: int, number of iterations for HBAC
        min_cluster_size: int, minimum cluster size
        seed: int, random seed
    """
    # set the seed
    set_seed(seed)

    # Initialize the HBAC model
    hbac = BiasAwareHierarchicalKMeans(n_iter=n_iter_hbac, min_cluster_size=min_cluster_size) 

    # Fit on training set
    hbac.fit(X_train, y_train)

    return hbac




def compute_diff_hbac(X_train, X_val, y_train, y_val, max_k, n_iter_hbac, min_cluster_size, seed, fit_train=True, target_col='y'):
    """
    Compute the difference in bias for each cluster

    Args:
        X_train: np.array, shape (N, d)
        X_val: np.array, shape (N, d)
        y_train: np.array, shape (N,)
        y_val: np.array, shape (N,)
        max_k: int, maximum number of clusters
        n_iter_hbac: int, number of iterations for HBAC
        min_cluster_size: int, minimum cluster size
        seed: int, random seed
        fit_train: bool, whether to fit on a train set and predict on a validation set
    """
    # set the seed
    set_seed(seed)

    # define the model
    if target_col == 'y':
        m_train = y_train
        m_val = y_val
    else:
        if binary_y:
            model = make_pipeline(StandardScaler(), LogisticRegression())
        else:
            model = make_pipeline(StandardScaler(), LinearRegression())
        
        # Fit the model on the training set
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val) # predicted labels on the validation set
        y_pred_train = model.predict(X_train) # predicted labels on the training set

        # if binary_y is True, then define target via binary metrics
        if binary_y:
            m_val = METRICS_BINARY[target_col](y_val, y_pred_val)
            m_train = METRICS_BINARY[target_col](y_train, y_pred_train)
        # Otherwise, define target via continuous metrics
        else:
            m_val = METRICS_CONTINUOUS[target_col](y_val, y_pred_val)
            m_train = METRICS_CONTINUOUS[target_col](y_train, y_pred_train)
        
    
    # Fit the HBAC model on the training set
    hbac = fit_hbac(X_train, m_train, n_iter_hbac, min_cluster_size, seed)

    # Fit on training set, predict on validation set
    if fit_train:
        labels = hbac.predict(X_val)
    # Else, fit/predict on the same set
    else:
        labels = hbac.labels_

    
    # create a array of nan to store the diffs
    diffs = np.full(max_k, np.nan)

    # Compute the bias, per label
    for l in np.unique(labels):

        # define cluster 1 (with label) and cluster 0 (~label)
        idx = labels == l
        
        # define cluster 1 (with label) and cluster 0 (~label)
        c1, c0 = m_val[idx], m_val[~idx]

        # Compute the difference in means
        diff_l = c1.mean() - c0.mean()

        # Store the difference
        diffs[l] = diff_l
    
    return diffs


def compute_diff_per_perm(n_perm, X_train, X_val, y_train, y_val, max_k, n_iter_hbac, min_cluster_size, seed, fit_train=True, conditional_on_k=False, target_col='y'):
    """
    Compute the difference in bias for each cluster, for n_perm permutations of y

    Args:
        n_perm: int, number of permutations
        X: np.array, shape (N, d)
        y: np.array, shape (N,)
        max_k: int, maximum number of clusters
        n_iter_hbac: int, number of iterations for HBAC
        min_cluster_size: int, minimum cluster size
        seed: int, random seed
        fit_train: bool, whether to fit on a train set and predict on a validation set
        val_frac: float, fraction of the data to use as validation set
    """

    # Compute the difference in bias for each cluster, for n_perm permutations of y
    diffs_perm = np.full((n_perm, max_k), np.nan)
    i=0

    # loop over each permutation
    for X_train_perm, X_val_perm, y_train_perm, y_val_perm in null_X_y(X_train, X_val, y_train, y_val, n_perm=n_perm, perm=True):
        diff_i = compute_diff_hbac(X_train_perm, X_val_perm, y_train_perm, y_val_perm, max_k, n_iter_hbac, min_cluster_size, seed, fit_train=fit_train, target_col=target_col)
        diffs_perm[i] = diff_i
        i += 1

    # remove rows if there are nan values
    if conditional_on_k:
        diffs_perm = diffs_perm[~np.isnan(diffs_perm[:, 1:]).any(axis=1)]

    return diffs_perm

def calc_p_value(diffs, diff, alternative='two-sided'):
    """
    Calculate the p-value for diff based on an empirical distribution of diffs.
    
    Args:
        diffs: np.array, shape (n_perm, )
        diff: float, difference in means

    """

    if alternative == 'two-sided':
        p_value = 2 * min(np.sum(diffs >= diff) / len(diffs), 
                         np.sum(diffs <= diff) / len(diffs))
    elif alternative == 'less':
        p_value = np.sum(diffs <= diff) / len(diffs)
    elif alternative == 'greater':
        p_value = np.sum(diffs >= diff) / len(diffs)
    else:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")
    return p_value


def compute_statistics(X, target, idx, bonf_correct, n_clust, bootstrap_perm, diffs_perm=None):
    """Compute statistics for each cluster/feature."""
    c1, c0 = target[idx], target[~idx]

    # if bootstrap_perm is True, calculate the diff and check the p-value via the null distribution
    if bootstrap_perm:

        # calculate the difference in means
        diff =  c1.mean() - c0.mean()

        # calculate the p-value
        p_clust = calc_p_value(diffs_perm, diff)
    
    else:
        # calculate the p-value using a t-test
        _, p_clust = ttest_ind(c1, c0, equal_var=False) 
    
    # apply bonferroni correction
    if bonf_correct:
        p_clust = p_clust * n_clust # Bonferroni correction - multiply by number of clusters

    diff_clust = c1.mean() - c0.mean() # Difference in means
    
    p_feat, diff_feat = [], []
    for ii in range(X.shape[1]):
        X_ = X[:, ii]
        c1, c0 = X_[idx], X_[~idx]

        should_continue = check_before_test(c0, c1)
        if should_continue:
            continue
        
        # calculate the p-value using a t-test
        _, p = ttest_ind(c1, c0)
        
        # if bonf_correct is True, multiply the p-value by the number of features
        if bonf_correct:
            p = p * X.shape[1]

        p_feat.append(p)
        diff = c1.mean() - c0.mean()
        diff_feat.append(diff)

    return p_clust, diff_clust, p_feat, diff_feat


def simulate_n_experiments(n_sims, parallel, method, K, N, y_dgp, x_dgp,  d, binary_y=False, fit_train=True, n_iter_hbac=10, min_cluster_size=5, val_frac=0.5, bonf_correct=True, target_col='y', bootstrap_perm=False, n_perm=1000, n_jobs=4):
    # if parallel is True, run the experiments in parallel
    if parallel: 
        results = Parallel(n_jobs=n_jobs)(delayed(simulate_experiment)(method, K, N, y_dgp, x_dgp,  d, seed=i, binary_y=binary_y, fit_train=fit_train, n_iter_hbac=n_iter_hbac, min_cluster_size=min_cluster_size, val_frac=val_frac, bonf_correct=bonf_correct, target_col=target_col, bootstrap_perm=bootstrap_perm, n_perm=n_perm) for i in range(n_sims))
    
    # otherwise, run the experiments sequentially
    else:
        results = [simulate_experiment(method, K, N, y_dgp, x_dgp,  d, seed=i, binary_y=binary_y, fit_train=fit_train, n_iter_hbac=n_iter_hbac, min_cluster_size=min_cluster_size, val_frac=val_frac, bonf_correct=bonf_correct, target_col=target_col) for i in range(n_sims)]
    

    # combine the results in a dataframe
    results_clust = pd.concat([r[0] for r in results if r is not None], axis=0)
    results_feat = pd.concat([r[1] for r in results if r is not None], axis=0)
    avg_missing = np.mean([r[2] for r in results if r is not None])
    print('Avg. number of clusters skipped per experiment: {}'.format(avg_missing))

    return results_clust, results_feat





def simulate_experiment(method, K, N, y_dgp, x_dgp,  d, seed, binary_y=False, fit_train=True, target_col='y', n_iter_hbac=10, min_cluster_size=5, val_frac=0.5, bonf_correct=True, bootstrap_perm=False, n_perm=1000):

    # simulate the outcome of the hbac
    out = simulate_hbac(method, K, N, y_dgp, x_dgp,  d, seed, binary_y=binary_y,  fit_train=fit_train, target_col=target_col, n_iter_hbac=n_iter_hbac, min_cluster_size=min_cluster_size, val_frac=val_frac)

    # if the outcome is None, return None
    if out is None:
        return None
    
    # otherwise, return the outcome
    X_val, X_train, m_val, m_train, labels, y_val, y_train = out

    # if fit_train is True, use m_val as the target variable
    if fit_train:
        target = m_val
    else:
        target = m_train

    # check: if the number of observations is too small, return None
    if X_val.shape[0] <= 5:
        return None
    
    # define the results
    n_clust = np.unique(labels).size
    results_clust = defaultdict(list)
    results_feat = defaultdict(list)

    # get the N total
    N = X_train.shape[0]

    # define the parameters to save
    params_ = list(zip(
        ['method', 'target_col', 'K', 'N', 'y_dgp', 'x_dgp', 'd', 'binary_y',  'fit_train', 'n_iter_hbac', 'min_cluster_size', 'val_frac', 'bonf_correct', 'bootstrap_perm', 'n_perm'],
        [method, target_col, K, N, y_dgp, x_dgp, d, binary_y, fit_train, n_iter_hbac, min_cluster_size, val_frac, bonf_correct, bootstrap_perm, n_perm]
    ))


    # if bootstrap_perm is True, compute the null distribution of differences
    if bootstrap_perm:
        diffs_perm = compute_diff_per_perm(n_perm, X_train, X_val, y_train, y_val,
                               n_clust, n_clust-1, min_cluster_size, seed, fit_train=fit_train, conditional_on_k=True, target_col=target_col)
        
        #print('Avg. diff in bias: {}'.format(np.nanmean(diffs_perm, axis=0)))
        
    else:
        diffs_perm = None

    # loop over each cluster
    count_missing = 0
    for l in np.unique(labels):

        # define cluster 1 (with label) and cluster 0 (~label)
        idx = labels == l
        c1, c0 = target[idx], target[~idx]

        should_continue = check_before_test(c0, c1)

        if should_continue:
            count_missing += 1
            continue
  

        # compute the statistics for each cluster/feature
        p_clust, diff_clust, p_feat, diff_feat = compute_statistics(X_val, target, idx, bonf_correct, n_clust, bootstrap_perm, diffs_perm[ :, l] if bootstrap_perm else None)
        results_clust['iter'].append(seed)
        results_clust['cluster_nr'].append(l)
        results_clust['p_clust'].append(p_clust)
        results_clust['diff_clust'].append(diff_clust)
        results_clust['size_clust'].append(idx.sum())


        # save the results in the dictionary
        for p_name_, p_ in params_:
            results_clust[p_name_].append(p_)

        # save the feature stats separately
        for i_feat, (p_feat_, diff_feat_) in enumerate(zip(p_feat, diff_feat)):
            results_feat['feat_nr'].append(i_feat)
            results_feat['p_feat'].append(p_feat_)
            results_feat['diff_feat'].append(diff_feat_)
            results_feat['size_feat'].append(idx.sum())

            for p_name, p_ in params_:
                results_feat[p_name].append(p_)

    
    # save the results in a dataframe
    results_clust = pd.DataFrame(results_clust)
    results_feat = pd.DataFrame(results_feat)

    return results_clust, results_feat, count_missing




if __name__ == '__main__':

    from params import n_sims
    from params import params

    EXPERIMENT_NAME='dgp'
    PARALLEL=True

    results_clust, results_feat = [], []
    for params in tqdm(params):

        # Get the parameters
        method, K, N, y_dgp, x_dgp, d, binary_y,  fit_train, n_iter_hbac, min_cluster_size, val_frac, bonf_correct, target_col, bootstrap_perm, n_perm = params
        print('Getting results for: method = {}, K = {}, N = {}, y_dgp = {}, x_dgp = {}, d = {}, binary_y = {},  fit_train = {}, n_iter_hbac = {}, min_cluster_size = {}, val_frac = {}, bonf_correct = {}, target_col = {}, bootstrap_perm = {}, n_perm = {}'.format(
            method, K, N, y_dgp, x_dgp, d, binary_y, fit_train, n_iter_hbac, min_cluster_size, val_frac, bonf_correct, target_col, bootstrap_perm, n_perm
        ))

        # Check: certain combinations of parameters are not allowed
        # First, we cannot have bootstrap_perm=True, and method != hbac
        if bootstrap_perm and method != 'hbac':
            print('Skipping this iteration, as bootstrap_perm=True, and method != hbac')
            continue

     

        # Simulate the experiment 
        results_clust_, results_feat_ = simulate_n_experiments(n_sims, parallel=PARALLEL, method=method, K=K, N=N, y_dgp=y_dgp, x_dgp=x_dgp, d=d,  binary_y=binary_y,  fit_train=fit_train, n_iter_hbac=n_iter_hbac, min_cluster_size=min_cluster_size, val_frac=val_frac, bonf_correct=bonf_correct,  target_col=target_col, bootstrap_perm=bootstrap_perm, n_perm=n_perm)

        # Append the results
        results_clust.append(results_clust_)
        results_feat.append(results_feat_)
   

    results_clust = pd.concat(results_clust, axis=0)
    results_feat = pd.concat(results_feat, axis=0)
    f_out = Path(__file__).parent / 'results_clust_{}_check.csv'.format(EXPERIMENT_NAME)
    results_clust.to_csv(f_out, index=False)
    f_out = Path(__file__).parent / 'results_feat_{}_check.csv'.format(EXPERIMENT_NAME)
    results_feat.to_csv(f_out, index=False)