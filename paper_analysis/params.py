from itertools import product






# define the parameters
n_sims = 1000
method = ['hbac']
target_col = ['y','y_pred']
K = [ 5 ]
N = [1000]
y_dgp = ['constant', 'linear']
x_dgp = ['random']
d = [2]
binary_y = [False]
fit_train = [True, False]
n_iter_hbac = ['known_clusters']
min_cluster_size = [5]
val_frac = [0.5]
bonf_correct = [True, False]
bootstrap_perm =[True, False]
n_perm = [1000]



# create the parameter grid
params = list(product(method, K, N, y_dgp, x_dgp,  d, binary_y, fit_train, n_iter_hbac, min_cluster_size, val_frac, bonf_correct, target_col, bootstrap_perm, n_perm))