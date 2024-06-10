"""Provides a default dataset with a healthcare focus for users to use."""
import fairlearn.datasets as data


def load_default_dataset():
    """
    Fetch a default healthcare dataset for use in x, y format.

    This function loads the diabetes hospital dataset from Msft's Fairlearn package and returns the x and y as pandas.
    The healthcare dataset provides data as to if a patient will be readmitted to hospital because of their diabetes.

    Parameters
    ----------
    None

    Returns
    -------
    diabetes_dataset_x: pandas.core.frame.DataFrame
        The features from the diabetes hospital dataset.
    diabetes_dataset_y: pandas.core.series.Series
        The target label (true label) for the diabetes hospital dataset which is readmit_30_days.
        One could use the variable readmit_binary as the target instead.

    See Also
    --------
    Dataset details: https://fairlearn.org/main/user_guide/datasets/diabetes_hospital_data.html

    Example
    --------
    >>> x, y = load_default_dataset()
    """
    print('Note: for the  unsupervised bias detection tool to work, a predictions column needs to be added. This column'
          'should be placed in between the features or x column and the y column so that it is the second to last '
          'column in a dataframe uploaded.')
    diabetes_dataset_x, diabetes_dataset_y = data.fetch_diabetes_hospital(return_X_y=True)
    return diabetes_dataset_x, diabetes_dataset_y