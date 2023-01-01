import sys
import random
import numpy as np
import pandas as pd
import seaborn as sns   
import pingouin as pg
import scipy.stats as stats

# matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import collections  as mc

# sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def init_GermanCredit_dataset(raw_data, features, with_errors=True, just_features=True, scale_features=True, with_classes=True):
    """ Initializing dataset: scaling features, adding new columns which are required for HBAC """

    new_data = raw_data.copy(deep=True)

    to_scale = new_data.drop(['predicted_class', 'true_class', 'errors', 'FP_errors', 'FN_errors'], axis=1).columns
    new_data[to_scale] = StandardScaler().fit_transform(features[to_scale])

    new_data['clusters'] = 0
    new_data['new_clusters'] = -1
    return new_data

def init_dataset(raw_data, features):
    """ Initializing dataset: scaling features, adding new columns which are required for HBAC """

    # copy dataframe
    new_data = raw_data.copy(deep=True)

    # only scale features
    to_scale = new_data.drop(['tweet','predicted_class', 'true_class', 'errors', 'FP_errors', 'FN_errors'], axis=1).columns
    new_data[to_scale] = StandardScaler().fit_transform(features[to_scale])
    new_data = new_data.drop(['tweet'], axis=1)

    # initialize clustering parameters
    new_data['clusters'] = 0
    new_data['new_clusters'] = -1
    
    return new_data
    
def bias(results, metric):
    ''' Return accuracy, FP rate or FN rate of dataframe '''
    
    if metric == 'Accuracy':
        correct = results.loc[results['errors'] == 0]
        acc = len(correct)/len(results)
        return acc
    if metric == 'FP':
        FPs = results.loc[(results['predicted_class'] == 1) & (results['true_class'] == 0)]
        Ns = results.loc[(results['true_class'] == 0)]
        if Ns.shape[0] != 0 :
            FP_rate = len(FPs)/len(Ns)
            return 1-FP_rate
        else:
            return 1
    if metric == 'FN':
        FNs = results.loc[(results['predicted_class'] == 0) & (results['true_class'] == 1)]
        Ps = results.loc[(results['true_class'] == 1)]
        if Ps.shape[0] != 0 :
            FN_rate = len(FNs)/len(Ps)
            return 1-FN_rate
        else:
            return 1

def bias_acc(data, metric, cluster_id, cluster_col):
    ''' Bias := bias metric of the selected cluster - bias metric of the remaining clusters '''
    cluster_x = data.loc[data[cluster_col] == cluster_id]
    if len(cluster_x) ==0:
        print("This is an empty cluster", cluster_id)
    remaining_clusters = data.loc[data[cluster_col] != cluster_id]
    if len(remaining_clusters) == 0:
        print("This cluster is the entire dataset", cluster_id)
    return bias(cluster_x, metric) - bias(remaining_clusters, metric)

def get_max_bias(fulldata, metric, function=bias_acc):
    ''' Calculates the highest negative bias of the newly introduced clusters '''
    max_bias = -999999
    for cluster_number in fulldata['new_clusters'].unique():
        current_bias = (function(fulldata, metric, cluster_number, "new_clusters"))
        if current_bias < max_bias:
            print('current bias: ', current_bias)
            print('max abs bias: ', max_bias)
            max_bias = current_bias
    return max_bias

def get_max_bias_cluster(fulldata, metric, function=bias_acc):
    ''' Identifies cluster linked to the highest bias of the newly introduced clusters '''
    max_bias = 100
    min_bias = -100
    best_cluster = -2
    for cluster_number in fulldata['clusters'].unique():
        current_bias = (function(fulldata, metric, cluster_number, "clusters"))
        print(f"{cluster_number} has bias {current_bias}")
        
        # Accuracy
        if metric == 'Accuracy':
            if current_bias < max_bias:
                max_bias = current_bias
                best_cluster = cluster_number

        # FP/FN
        if metric == 'FP' or metric == 'FN':
            if current_bias > min_bias:           
                min_bias = current_bias
                best_cluster = cluster_number

    return best_cluster

def get_min_cluster_size(data):
    ''' Size of smallest new cluster '''
    min_cluster_size = len(data)
    for i in data['new_clusters'].unique():
        # exclude the cluster -1 from being seen as a cluster, since it contains outliers
        if i == -1:
            continue
        size = len(data.loc[data['new_clusters']==i])
        if size < min_cluster_size:
            min_cluster_size = size
    return min_cluster_size

def get_next_cluster(data, metric):
    ''' Identifies cluster number with the highest variance. The variance is calculated based on the error metric of each cluster. The cluster with the highest variance will be selected as splitting cluster '''
    n_cluster = max(data['clusters'])
    highest_variance = -1
    cluster_number = 0

    for i in data['clusters'].unique():
        if (i == -1):
            continue
        cluster_i = data.loc[data['clusters'] == i]
        if metric == 'Accuracy':
            variance_cluster = np.var(cluster_i['errors'])
        if metric == 'FP':
            variance_cluster = np.var(cluster_i['FP_errors'])
        if metric == 'FN':
            variance_cluster = np.var(cluster_i['FN_errors'])
        
        if variance_cluster > highest_variance:
            highest_variance = variance_cluster
            cluster_number = i

    return cluster_number

def calculate_variance(data, metric):
    ''' Determines variance for a dataframe. '''
    variance_list_local = []
    for j in data['clusters'].unique():
        average_bias = bias(data, metric)
        bias_clus = bias_acc(data, metric, j, 'clusters') 
        variance_list_local.append(bias_clus) 
    variance = np.var(variance_list_local)
    return variance

def get_random_cluster(clusters):
    ''' Identifies value of a random cluster '''
    result = -1
    while (result == -1):
        result = random.randint(0, len(clusters.unique()))
    return result

def HBAC_bias_scan(df, metric, split_cluster_size, acc_cluster_size, clustering_paramaters):
    iterations_max = 20
    x = 0 # initial cluster number
    initial_bias = 0
    variance_list = []
    average_bias = bias(df, metric)
    minimal_splittable_cluster_size = split_cluster_size
    minimal_acceptable_cluster_size = acc_cluster_size
    print(f"bias {metric} is: ", average_bias) 

    for i in range(1, iterations_max):
        if i != 1:

            # calculate variance for cluster
            variance_list.append(calculate_variance(df, metric)) 

        df['new_clusters'] = -1
        candidate_cluster = df.loc[df['clusters'] == x]

        if len(candidate_cluster) < minimal_splittable_cluster_size:
            x = get_random_cluster(df['clusters'])
            continue

        # k-means clustering 
        kmeans_algo = KMeans(**clustering_paramaters).fit(candidate_cluster.drop(['clusters', 'new_clusters', 'predicted_class', 'true_class', 'errors', 'FP_errors', 'FN_errors'], axis=1))

        candidate_cluster['new_clusters'] = pd.DataFrame(kmeans_algo.predict(candidate_cluster.drop(['clusters', 'new_clusters', 'predicted_class', 'true_class', 'errors', 'FP_errors', 'FN_errors'], axis=1)),index=candidate_cluster.index) 
        df['new_clusters'] = candidate_cluster['new_clusters'].combine_first(df['new_clusters'])

        # find discriminated clusters
        max_bias = get_max_bias(df, metric) 
        min_new_size = get_min_cluster_size(df)
        
        if (max_bias <= initial_bias) & (min_new_size > minimal_acceptable_cluster_size):
            # Add new cluster
            n_cluster = max(df['clusters'])
            df['clusters'][df['new_clusters'] == 1] =  n_cluster + 1

            x = get_next_cluster(df, metric)
            initial_bias = max_bias
        else:
            x = get_random_cluster(df['clusters'])

    print('done')
    return(df)

def stat_df(df, discriminated_cluster, not_discriminated):

    # finding difference
    difference = (discriminated_cluster.mean()) - (not_discriminated.mean()) 
    diff_dict = difference.to_dict()
    
    # unscaling the discriminated cluster
    unscaled_discriminated = df.loc[discriminated_cluster.index, :]

    # unscaled other data
    unscaled_remaining = df.drop(discriminated_cluster.index)
    
    # statistical testing
    welch_dict = {}
    CI_dict_left = {}
    CI_dict_right = {}

    features = [col for col in df.columns.tolist() if col not in ['tweet_id1','scaled_errors','predicted_class','true_class','errors', 'FP_errors', 'FN_errors','clusters','new_clusters']]

    for i in features:
        welch_i = stats.ttest_ind(unscaled_discriminated[i], unscaled_remaining[i], equal_var=False)
        res = pg.ttest(unscaled_discriminated[i], unscaled_remaining[i], paired=False)

        # attach to dictionary
        welch_dict[i] = welch_i.pvalue
        CI_dict_left[i] = res["CI95%"][0][0]
        CI_dict_right[i] = res["CI95%"][0][1]
        
    # store results in dataframe
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    cluster_analysis_df = pd.DataFrame([diff_dict, welch_dict, CI_dict_left, CI_dict_right]).T
    cluster_analysis_df.columns = ['difference','p-value','[0.025','0.975]']
    cluster_analysis_df = cluster_analysis_df.sort_values('p-value',ascending=[True])
    n_rows = cluster_analysis_df.shape[0]

    # Get errors; (coef - lower bound of conf interval)
    cluster_analysis_df['errors'] = cluster_analysis_df['difference'] - cluster_analysis_df['[0.025']
    cluster_analysis_df = cluster_analysis_df.iloc[0:n_rows,]
    cluster_analysis_df['num'] = [int(i) for i in np.linspace(n_rows-1,0,n_rows)]

    cluster_analysis_df = cluster_analysis_df.reset_index()
    
    return(cluster_analysis_df)

def CI_plot(df, x_lim, feat_ls):
    '''
    Takes in results of Welch's t-test and returns a plot of 
    the coefficients with 95% confidence intervals.
    '''   
    n_rows = df.shape[0]

    # line segments
    lines_sign = []
    lines_non_sign = []
    index_ls = []
    i=n_rows
    for feat in feat_ls:
        k = df[df['index'] == feat].index[0]  
        p_value = df.iloc[k,2]
        if p_value <= 0.05:
            sub_ls_sign = []
            sub_ls_sign.append((df.iloc[k,3],i))
            sub_ls_sign.append((df.iloc[k,4],i))    
            lines_sign.append(sub_ls_sign)
            index_ls.append((i,k))
            i-=1
        else:
            sub_ls_non_sign = []
            sub_ls_non_sign.append((df.iloc[k,3],i))
            sub_ls_non_sign.append((df.iloc[k,4],i))    
            lines_non_sign.append(sub_ls_non_sign)
            index_ls.append((i,k))
            i-=1

    fig, ax = plt.subplots(figsize=(10, 7))

    # Line to define zero on the x-axis
    ax.axvline(x=0, linestyle='--', color='black', linewidth=1)

    # line segments significant
    lc = mc.LineCollection(lines_sign, colors='steelblue', linewidths=10, alpha=0.75)
    ax.add_collection(lc)
    ax.autoscale()

    # line segments non-significant
    lc = mc.LineCollection(lines_non_sign, colors='steelblue', linewidths=10, alpha=0.25)
    ax.add_collection(lc)
    ax.autoscale()

    # title and axes
    plt.title('Cluster difference 95% confidence interval',fontsize=24)
    
    # font size axes
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # x-axis
    ax.set_xlabel('Difference in means',fontsize=22)
    ax.set_xlim(x_lim)
    xlims = ax.get_xlim()
    
    # annotate x-axis
    ax.annotate('Cluster mean lower\nthan rest of dataset',xy=(xlims[0],-0.1),xytext=(xlims[0],-0.5), ha='center', annotation_clip=False, fontsize=14, style='italic')
    ax.annotate('Cluster mean higher\nthan rest of dataset',xy=(xlims[1],-0.1),xytext=(xlims[1],-0.5), ha='center', annotation_clip=False, fontsize=14, style='italic')
    
    # y-axis
    columns = feat_ls
    ax.set_yticklabels(['']+columns[::-1])

    # scatter plot
    idx_ls = [i for (i,k) in index_ls]
    scatter_ls = [df.iloc[k,1] for (i,k) in index_ls]
    ax.scatter(y=idx_ls,
               marker='o', s=250, edgecolors='none', linewidth=2,
               x=scatter_ls, color='steelblue')
    
    # legend
    legend_elements = [Line2D([0], [0], color='steelblue', alpha=0.75, lw=10, label='Significant'),
                       Line2D([0], [0], color='steelblue', alpha=0.25, lw=10, label='Not significant')]
    ax.legend(handles=legend_elements, loc='best', fontsize=16)

    return plt.show()

def pca_plot(data):
    """ PCA dimensionality reduction to display identified clusters as scatterplot. """
    
    pca_features = data.drop(['predicted_class', 'true_class', 'errors', 'FP_errors', 'FN_errors', 'clusters', 'new_clusters'], axis=1)
    other_features = data[['predicted_class', 'true_class', 'errors', 'FP_errors', 'FN_errors', 'clusters', 'new_clusters']]
    
    df = pd.DataFrame(pca_features)
    pca = pd.DataFrame(PCA(n_components=2).fit_transform(df), index=df.index)
    temp_dataset = pca.join(other_features, how='left')
    temp_dataset.rename( columns={0 :'PCA - 1st'}, inplace=True )
    temp_dataset.rename( columns={1 :'PCA - 2nd'}, inplace=True )

    scatterplot = sns.scatterplot(data=temp_dataset, x='PCA - 1st', y='PCA - 2nd', hue="clusters", size='errors', sizes=(150, 30), palette="Set1")
    scatterplot.set_title('HBAC bias scan (k-means) on AI classifier')
    lgd = scatterplot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
    plt.show()
#     plt.savefig('./test.png', bbox_extra_artists=(lgd,), bbox_inches='tight')