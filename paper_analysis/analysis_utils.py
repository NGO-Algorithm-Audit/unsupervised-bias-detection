import seaborn as sns
import matplotlib.pyplot as plt

def plot_diff_in_bias(df, target_col_map={'y': r'$y$', 'y_pred': r'$\hat{y}$', 'err': r'Error'}, color_set='Reds', ax=None):

    # if no ax is provided, create a new figure
    if ax is None:
        fig, ax = plt.subplots()
    
    # make a copy of the df for the plot
    df_p = df.copy()

    # create the color palette for seaborn
    color_palette = sns.color_palette(color_set, n_colors=df['cluster_nr'].nunique())

    # change the target_col values based on a dictionary
    df_p['target_col'] = df_p['target_col'].map(target_col_map)

    # Map the cluster numbers to have +1
    df_p['cluster_nr'] = df_p['cluster_nr'] + 1

    # create the plot
    sns.barplot(data=df_p, x='target_col', y='diff_clust', hue='cluster_nr', palette=color_palette, errorbar = ('ci', 95), ax=ax)

    # set the labels
    ax.set_xlabel('Bias metric')
    ax.set_ylabel('Difference in Bias')

    # define the title of the legend
    ax.legend(title='Cluster')

    return ax

def plot_grid_of_bias_diffs(df, K_values, N_values, target_col_map={'y': r'$y$', 'y_pred': r'$\hat{y}$', 'err': r'Error'}, color_set='Reds'):
    fig, axes = plt.subplots(len(N_values), len(K_values), figsize=(5 * len(K_values), 5 * len(N_values)), sharey=True)

    for i, N in enumerate(N_values):
        for j, K in enumerate(K_values):
            # Create a modified version of the dataframe for each combination of K and N
            df_mod = df[(df['K'] == K) & (df['N'] == N)].copy()
            
            # Create the plot using the existing function
            plot_diff_in_bias(df_mod, target_col_map, color_set, ax=axes[i, j])
            
            # Set the title for each subplot
            axes[i, j].set_title(r'$K={}$, $N={}$'.format(K, N))
    
    # Adjust layout
    plt.tight_layout()

    return fig





