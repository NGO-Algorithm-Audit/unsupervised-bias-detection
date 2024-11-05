

import seaborn as sns



def plot_diff_in_bias(df, target_col_map={'y': r'$y', 'y_pred': r'$\hat{y}$', 'err': r'Error'},  color_set='Reds'):

    # make a copy of the df for the plot
    df_p = df.copy()

    # create the color palette for seaborn
    color_palette = sns.color_palette(color_set, n_colors=df['cluster_nr'].nunique())

    # change the target_col values based on a dictionary
    df_p['target_col'] = df_p['target_col'].map(target_col_map)

    # Map the cluster numbers to have +1
    df_p['cluster_nr'] = df_p['cluster_nr'] + 1

    # create the plot
    g = sns.catplot(df_p, x='target_col', y='diff_clust', hue='cluster_nr', kind='bar', palette=color_palette, 
                     errorbar=("se", 1.96))

    # set the labels
    g.set_axis_labels('Bias metric', 'Difference in Bias')

    # define the title of the legend
    g._legend.set_title('Cluster')

    return g