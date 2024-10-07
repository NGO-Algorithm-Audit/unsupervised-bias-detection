import numpy as np
import pandas as pd
from scipy.stats import norm, ks_2samp

class GaussianCopulaSynthesizer:
    def __init__(self):
        self.means = None
        self.cov_matrix = None
        self.scaler = None
        self.data_marginals = None

    def fit(self, data):
        """
        Fit the Gaussian Copula model to the given data.
        """
        # Step 1: Store data marginals (quantiles for each feature)
        self.data_marginals = []
        for col in data.columns:
            sorted_data = np.sort(data[col])
            quantiles = np.linspace(0, 1, len(sorted_data))
            self.data_marginals.append((sorted_data, quantiles, col))

        # Step 2: Convert data to normal distribution using CDF (Gaussianization)
        uniform_data = data.rank(pct=True)  # Get percentile rank for each column (empirical CDF)
        gaussian_data = norm.ppf(uniform_data)  # Convert uniform to standard normal

        # Step 3: Fit a multivariate Gaussian to the normalized data
        self.means = gaussian_data.mean(axis=0)
        self.cov_matrix = np.cov(gaussian_data, rowvar=False)

    def sample(self, n_samples):
        """
        Generate synthetic data using the fitted Gaussian Copula model.
        """
        # Step 1: Sample from the multivariate normal distribution
        synthetic_gaussian = np.random.multivariate_normal(self.means, self.cov_matrix, n_samples)

        # Step 2: Convert back to uniform distribution using CDF (normal -> uniform)
        synthetic_uniform = norm.cdf(synthetic_gaussian)

        # Step 3: Map uniform data back to the original marginals
        synthetic_data = pd.DataFrame()
        for i, (sorted_data, quantiles, col) in enumerate(self.data_marginals):
            synthetic_data[col] = np.interp(synthetic_uniform[:, i], quantiles, sorted_data)

        return synthetic_data
    

def evaluate_distribution(real_data, synthetic_data):
    """
    Compare the distribution of each column in the real and synthetic data using
    the Kolmogorov-Smirnov (KS) test.
    """
    results = {}
    for column in real_data.columns:
        real_col = real_data[column].dropna()
        synthetic_col = synthetic_data[column].dropna()

        # Perform the KS test
        ks_stat, p_value = ks_2samp(real_col, synthetic_col)

        # Store the result
        results[column] = {'ks_stat': ks_stat, 'p_value': p_value}
    return results

def evaluate_correlations(real_data, synthetic_data):
    """
    Compare the pairwise correlation matrices of the real and synthetic data.
    """
    real_corr = real_data.corr()
    synthetic_corr = synthetic_data.corr()

    # Compute the difference between the correlation matrices
    corr_diff = np.abs(real_corr - synthetic_corr)
    return corr_diff.mean().mean()  # Average correlation difference

def run_diagnostic(real_data, synthetic_data, target_column):
    """
    Run diagnostics on synthetic data by evaluating distribution, correlations, and
    classification model performance.
    """
    # Step 1: Evaluate distributions
    distribution_results = evaluate_distribution(real_data, synthetic_data)

    # Step 2: Evaluate correlations
    correlation_diff = evaluate_correlations(real_data, synthetic_data)

    # Aggregate results
    diagnostics = {
        'distribution_results': distribution_results,
        'correlation_diff': correlation_diff
    }

    return diagnostics