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

def set_seed(seed):
    np.random.seed(seed)


METRICS_BINARY = {
    'y': lambda y, y_pred: y,
    'y_pred': lambda y, y_pred: y_pred,
    'fp': lambda y, y_pred: np.logical_and(y == 0, y_pred == 1).astype(int),
    'fn': lambda y, y_pred: np.logical_and(y == 1, y_pred == 0).astype(int),
    'err': lambda y, y_pred: (y != y_pred).astype(int)
}

METRICS_CONTINUOUS = {
    'y': lambda y, y_pred: y,
    'y_pred': lambda y, y_pred: y_pred,
    'errr': lambda y, y_pred: (y - y_pred)**2
}


def simulate_synthetic_data( K, N_k, y_dgp, x_dgp, d, seed, binary_y=False):

    """
    Follows the DGP from Misztal-Radecka & Indurkhya

    Arguments:
        - K: either 
            int: number of segments
            str: 'random', then uniformly sample K from [2  10]
        - N_k: either
            int: number of observations per segment
            str: 'random', then uniformly sample N_k from [10  200]
        - y_DGP: str, one of the following options
            random: M \sim N(mu_k, sigma_k), where mu_k, sigma_k are sampled from [0  1]
            constant: M \sim N(0, 1) for all segments
            linear: M \sim N(0, 1) + 0.1 * k, e.g. an increasing trend per segment
        - x_DGP: str, one of the following options
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
    if N_k == 'random':
        N_k_values = np.random.randint(10, 200, K)
    else:
        N_k_values = [N_k] * K
    
    # create matrix to store the data with shape (N, d+1, K)
    data = np.zeros((np.sum(N_k_values), d+2))
    i = 0

    # loop over each segment
    for k in range(K):
        
        # get the number of observations for this segment
        n_k = N_k_values[k]

        # update the index
        i += n_k

        # first, get the dgp for the target variable y
        # if y_dgp is random, sample mu_k, sigma_k from [0  1]
        if y_dgp == 'random':
            mu_k, sigma_k = np.random.uniform(0, 1, 2)
        
        # generate the features as a matrix X
        elif y_dgp == 'constant':
            mu_k, sigma_k = 0, 1
        
        # generate the features as a matrix X
        elif y_dgp == 'linear':
            mu_k, sigma_k = 0, 1
            mu_k += 0.1 * k

        # now generate the target variable y
        y_k = np.random.normal(mu_k, sigma_k, n_k)

        # if binary_y is True, then make the target variable binary
        if binary_y:
            y_k = np.where(y_k > 0, 1, 0)

        # then, define the DGP for the features X
        if x_dgp == 'constant':
            mu_k = np.zeros(d)
            sigma_k = np.eye(d)
        
        # generate the features as a matrix X
        X_k = np.random.multivariate_normal(mu_k, sigma_k, n_k)

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

def simulate_hbac(method, target_col, K, N_k, y_dgp, x_dgp, d, seed, binary_y=False, randomize_y=False, fit_train=True, n_iter_hbac=10, min_cluster_size=5, val_frac=0.8):

    # simulate the synthetic data
    result_sim = simulate_synthetic_data(K, N_k, y_dgp, x_dgp, d, seed, binary_y=binary_y)
    X = result_sim['data'][:, 1:-1]
    y = result_sim['data'][:, 0].flatten()
    k = result_sim['data'][: , -1].flatten()


    # if randomize_y is True, then shuffle the target variable
    if randomize_y:
        np.random.shuffle(y)

    # define the model
    if binary_y:
        model = make_pipeline(StandardScaler(), LogisticRegression())
        stratify = y
    else:
        model = make_pipeline(StandardScaler(), LinearRegression())
        stratify = None
      
    # Split in train and val set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_frac, stratify=stratify)

    # Fit the model on the training set
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val) # predicted labels on the validation set
    y_pred_train = model.predict(X_train) # predicted labels on the training set


    # if binary_y is True, then define target via binary metrics
    if binary_y:
        target_val = METRICS_BINARY[target_col](y_val, y_pred_val)
        target_train = METRICS_BINARY[target_col](y_train, y_pred_train)
    # Otherwise, define target via continuous metrics
    else:
        target_val = METRICS_CONTINUOUS[target_col](y_val, y_pred_val)
        target_train = METRICS_CONTINUOUS[target_col](y_train, y_pred_train)
    
    # obtain cluster labels
    if method in ['kmeans', 'kmeans_cv']:

        # define the cluster model
        set_seed(seed)
        cluster_model = KMeans(n_clusters=K)

        # fit the model on the training set
        if method == 'kmeans':
            cluster_model.fit(X_train)
            labels = cluster_model.predict(X_val)
        else:
            labels = cluster_model.fit_predict(X_val)
    
    # if method is hbac, use the BiasAwareHierarchicalKMeans
    elif method == 'hbac':
        set_seed(seed)
        hbac = BiasAwareHierarchicalKMeans(n_iter=n_iter_hbac, min_cluster_size=min_cluster_size) # 5 is the minimum

        # Fit on training set, predict on validation set
        if fit_train:
            hbac.fit(X_train, target_train)
            labels = hbac.predict(X_val)
        # Fit on validation set, predict on validation set
        else:
            hbac.fit(X_val, target)
            labels = hbac.labels_

    # if method is randomclusters, randomly assign cluster labels
    elif method == 'randomclusters':
        labels = np.random.choice(range(K), size=X_val.shape[0])
    else:
        raise ValueError(f"Not a known method ({method})")
    

    return X_val, X_train, target_val, target_train, labels


def check_before_test(c0, c1, min_samples=5):
    """Determines whether to do significance testing
    (don't do this with too few observations/constant data to avoid NaNs)."""
    if (c0.size < min_samples) or (c1.size < min_samples):
        print('Too few observations, skipping this cluster/feature.')
        return True

    if np.mean(c0) == np.mean(c1):
        print('Means are equal, skipping this cluster/feature.')
        return True
    
    if np.all(c0 == c0[0]) or np.all(c1 == c1[0]):
        print('Data is constant, skipping this cluster/feature.')
        return True

    return False


def compute_statistics(X, target, idx, bonf_correct, n_clust):
    """Compute statistics for each cluster/feature."""
    c1, c0 = target[idx], target[~idx]
    _, p_clust = ttest_ind(c1, c0, equal_var=False)
    
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
        
        _, p = ttest_ind(c1, c0)
        if bonf_correct:
            p = p * X.shape[1]

        p_feat.append(p)
        diff = c1.mean() - c0.mean()
        diff_feat.append(diff)

    return p_clust, diff_clust, p_feat, diff_feat


def simulate_n_experiments(n_sims, parallel, method, target_col, K, N_k, y_dgp, x_dgp, d, binary_y=False, randomize_y=False, fit_train=True, n_iter_hbac=10, min_cluster_size=5, val_frac=0.8, bonf_correct=True, n_jobs=4):

    # if parallel is True, run the experiments in parallel
    if parallel:
        results = Parallel(n_jobs=n_jobs)(delayed(simulate_experiment)(method, target_col, K, N_k, y_dgp, x_dgp, d, seed=i, binary_y=binary_y, randomize_y=randomize_y, fit_train=fit_train, n_iter_hbac=n_iter_hbac, min_cluster_size=min_cluster_size, val_frac=val_frac, bonf_correct=bonf_correct) for i in range(n_sims))
    
    # otherwise, run the experiments sequentially
    else:
        results = [simulate_experiment(method, target_col, K, N_k, y_dgp, x_dgp, d, seed=i, binary_y=binary_y, randomize_y=randomize_y, fit_train=fit_train, n_iter_hbac=n_iter_hbac, min_cluster_size=min_cluster_size, val_frac=val_frac, bonf_correct=bonf_correct) for i in range(n_sims)]
    

    # combine the results in a dataframe
    results_clust = pd.concat([r[0] for r in results if r is not None], axis=0)
    results_feat = pd.concat([r[1] for r in results if r is not None], axis=0)

    return results_clust, results_feat





def simulate_experiment(method, target_col, K, N_k, y_dgp, x_dgp, d, seed, binary_y=False, randomize_y=False, fit_train=True, n_iter_hbac=10, min_cluster_size=5, val_frac=0.8, bonf_correct=True):

    # simulate the outcome of the hbac
    out = simulate_hbac(method, target_col, K, N_k, y_dgp, x_dgp, d, seed, binary_y=binary_y, randomize_y=randomize_y, fit_train=fit_train, n_iter_hbac=n_iter_hbac, min_cluster_size=min_cluster_size, val_frac=val_frac)

    # if the outcome is None, return None
    if out is None:
        return None
    
    # otherwise, return the outcome
    X_val, X_train, target_val, target_train, labels = out

    # if fit_train is True, use target_val as the target variable
    if fit_train:
        target = target_val
    else:
        target = target_train

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
        ['method', 'target_col', 'K', 'N_k', 'N', 'y_dgp', 'x_dgp', 'd', 'binary_y', 'randomize_y', 'fit_train', 'n_iter_hbac', 'min_cluster_size', 'val_frac', 'bonf_correct'],
        [method, target_col, K, N_k, N, y_dgp, x_dgp, d, binary_y, randomize_y, fit_train, n_iter_hbac, min_cluster_size, val_frac, bonf_correct]
    ))

    # loop over each cluster
    for l in np.unique(labels):

        # define cluster 1 (with label) and cluster 0 (~label)
        idx = labels == l
        c1, c0 = target[idx], target[~idx]

        should_continue = check_before_test(c0, c1)
        if should_continue:
            print('Skipping this cluster because of too few data/constant data')
            continue
  

        # compute the statistics for each cluster/feature
        p_clust, diff_clust, p_feat, diff_feat = compute_statistics(X_val, target, idx, bonf_correct, n_clust)
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

    return results_clust, results_feat




if __name__ == '__main__':

    from params import n_sims
    from params import params

    results_clust, results_feat = [], []
    for params in tqdm(params):

        # Get the parameters
        method, target_col, K, N_k, y_dgp, x_dgp, d, binary_y, randomize_y, fit_train, n_iter_hbac, min_cluster_size, val_frac, bonf_correct = params
        print('Getting results for: method = {}, target_col = {}, K = {}, N_k = {}, y_dgp = {}, x_dgp = {}, d = {}, binary_y = {}, randomize_y = {}, fit_train = {}, n_iter_hbac = {}, min_cluster_size = {}, val_frac = {}, bonf_correct = {}'.format(
            method, target_col, K, N_k, y_dgp, x_dgp, d, binary_y, randomize_y, fit_train, n_iter_hbac, min_cluster_size, val_frac, bonf_correct
        ))

        # Simulate the experiment 
        results_clust_, results_feat_ = simulate_n_experiments(n_sims, parallel=True, method=method, target_col=target_col, K=K, N_k=N_k, y_dgp=y_dgp, x_dgp=x_dgp, d=d,  binary_y=binary_y, randomize_y=randomize_y, fit_train=fit_train, n_iter_hbac=n_iter_hbac, min_cluster_size=min_cluster_size, val_frac=val_frac)

        # Append the results
        results_clust.append(results_clust_)
        results_feat.append(results_feat_)

    results_clust = pd.concat(results_clust, axis=0)
    results_feat = pd.concat(results_feat, axis=0)
    f_out = Path(__file__).parent / 'results_clust.csv'
    results_clust.to_csv(f_out, index=False)
    f_out = Path(__file__).parent / 'results_feat.csv'
    results_feat.to_csv(f_out, index=False)