import numpy as np
import pandas as pd
import scipy.stats as stats

def diff_df(df, features, type=None, cluster1=None, cluster2=None):
    '''
    Creates difference dataframe, for numerical data only: 
    Takes dataframe of two clusters of interest and 
    computes difference in means, incl. 95% confidence
    interval. Default to analyze most deviating cluster 
    vs rest of the dataset, except specified otherwise.
    '''   

    # Cluster comparison (optional)
    if cluster1 != None and cluster2 != None:
        # Dataframes
        df1 = df[df['Cluster'] == cluster1]
        df2 = df[df['Cluster'] == cluster2]

    # Default (most biased vs rest of dataset)
    else:
        # Dataframes
        df1 = df[df['Cluster'] == 0]
        df2 = df[df['Cluster'] != 0]

    # Number of datapoints in clusters
    n_df1 = df1.shape[0]
    n_df2 = df2.shape[0]

    # Initialize dictionaries
    diff_dict = {}
    CI_dict = {}

    # range through features
    for feat in features:
        # Samples
        sample1 = df1[feat]
        sample2 = df2[feat]

        # numercial data
        if type == 'Numerical':
            # Mean per sample
            mean1 = np.mean(sample1)
            mean2 = np.mean(sample2)

            # Difference in sample means
            diff = mean1 - mean2

            # Store results in dict
            diff_dict[feat] = diff

        # categorical data
        else:
            # get all values for categorical feature
            freq1 = sample1.value_counts()
            freq2 = sample2.value_counts()

            # difference in sample freqs
            diff = freq1 - freq2

            # Store results in dict
            diff_dict[feat] = diff

        # # Standard deviation per sample
        # std1 = np.std(sample1, ddof=1)  # ddof=1 for sample standard deviation
        # std2 = np.std(sample2, ddof=1)

        # # Standard error of the difference
        # SE = np.sqrt((std1**2 / n_df1) + (std2**2 / n_df2))

        # # Degrees of freedom for the t-distribution
        # degree_fr = n_df1 + n_df2 - 2

        # # Determine the critical value (t-value) for a 95% confidence interval
        # t_critical = stats.t.ppf(1 - 0.025, degree_fr)  # 95% confidence -> alpha = 0.05, two-tailed

        # # Margin of error
        # ME = t_critical * SE

        # # Confidence intervals
        # lower_bound = diff - ME
        # upper_bound = diff + ME

        # # store confidence interval
        # CI_dict[feat] = (lower_bound, upper_bound)

        # store numerical results in dataframe
        if type == 'Numerical':
            # Store results in dataframe
            pd.set_option('display.float_format', lambda x: '%.5f' % x)
            diff_df = pd.DataFrame.from_dict(diff_dict, orient='index', columns=['Difference'])
            # diff_df.columns = ['lower CI', 'upper CI']
            # diff_df['diff sample means'] = diff_df.index.map(diff_dict)
            
        # store numerical results in dataframe
        else:
            # Store results in dataframe
            diff_df = pd.DataFrame()
            pd.set_option('display.float_format', lambda x: '%.5f' % x)

            # range through all values per feature and concatenate to dataframe
            for _, value in diff_dict.items():
                df_temp = pd.DataFrame(value)
                diff_df = pd.concat([diff_df,df_temp], axis=0,)

            # replace Nan with 0
            diff_df = diff_df.fillna(0)

            # rename columns
            diff_df.columns = ['Difference']   

    return(diff_df)