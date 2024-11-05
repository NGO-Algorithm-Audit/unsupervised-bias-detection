from itertools import product






# define the parameters
n_sims = 1000
method = ['hbac', 'randomclusters']
target_col = ['y','y_pred', 'err']
K = [5, 10, 50 ]
N = [1000, 10000]
y_dgp = ['constant', 'linear']
x_dgp = ['random']
d = [2, 10, 50]
binary_y = [True, False]
randomize_y = [False]
fit_train = [False, True]
n_iter_hbac = ['known_clusters']
min_cluster_size = [5]
val_frac = [0.5]
bonf_correct = [True, False]


# create the parameter grid
params = list(product(method, target_col, K, N, y_dgp, x_dgp, d, binary_y, randomize_y, fit_train, n_iter_hbac, min_cluster_size, val_frac, bonf_correct))