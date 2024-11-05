from itertools import product






# define the parameters
n_sims = 100
method = ['hbac', 'randomclusters']
target_col = ['y','y_pred', 'err']
K = [5]
N_k = [400]
y_dgp = ['constant']
x_dgp = ['constant']
d = [10]
binary_y = [True]
randomize_y = [False]
fit_train = [True]
n_iter_hbac = [5]
min_cluster_size = [5]
val_frac = [0.5]
bonf_correct = [False, True]


# create the parameter grid
params = list(product(method, target_col, K, N_k, y_dgp, x_dgp, d, binary_y, randomize_y, fit_train, n_iter_hbac, min_cluster_size, val_frac, bonf_correct))