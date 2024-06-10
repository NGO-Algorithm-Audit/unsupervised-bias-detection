"""Provides functions for testing dataset properties."""
import pandas as pd


# TODO: add functionality to complete checks if dealing with a numpy array instead of pandas


def __data_preprocessing(data):
    """
    Validate dataset is pandas and extract information about the dataset and returns that info in the form of variables.

    This private method checks the dataset is a pandas dataframe. It extracts the features, predictions, and true labels
    from the dataset and returns them.

    Parameters
    ----------
    data: pandas dataframe

    Returns
    -------
    features: pandas.core.series.Series
    predictions: pandas.core.series.Series
    true_labels: pandas.core.series.Series

    See Also
    --------
    More dataset details: https://fairlearn.org/main/user_guide/datasets/diabetes_hospital_data.html

    Example
    --------
    >>> x, preds, labels = __data_preprocessing(dataset)
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be of type pandas.DataFrame.")

    column_length = len(data.columns)
    features = data.iloc[:, column_length - 3]
    predictions = data.iloc[:, column_length-2]
    true_labels = data.iloc[:, column_length-1]
    return features, predictions, true_labels


def __check_numerical_x_y(features, predictions, true_labels):
    """
    Test that the x (features) and y (preds/labels) are numerical.

    Parameters
    ----------
    features: pandas.core.series.Series
    predictions: pandas.core.series.Series
    true_labels: pandas.core.series.Series

    Returns
    -------
    None

    Example
    --------
    >>> __check_numerical_x_y(features, preds, labels)
    """
    for i in range(len(features)):
        row = features[i]
        pred = str(predictions[i])
        true_lab = str(true_labels[i])
        for x in range(len(row)):
            # numerical x check
            if not str(row[x]).isnumeric():
                raise ValueError('Features must be numeric.')
        # numerical y check
        if not (pred.isnumeric() and true_lab.isnumeric()):
            raise ValueError('Labels and predictions must be numeric.')
    return


def __check_binary_class(predictions, true_labels):
    """
    Test that the predictions and true labels are binary in value (0 or 1).

    Parameters
    ----------
    predictions: pandas.core.series.Series
    true_labels: pandas.core.series.Series

    Returns
    -------
    None

    Example
    --------
    >>> __check_binary_class(preds, labels)
    """
    for i in range(len(predictions)):
        pred = str(predictions[i])
        true_lab = str(true_labels[i])
        if not ((pred == '0' or pred == '1') and (true_lab == '0' or true_lab == '1')):
            raise ValueError('Labels and predictions should be 0 or 1 for binary classification.')
    return


# Public method that runs the private functions to test 3 properties of the dataset
def run_checks(data):
    """
    Test all the property tests for the dataset by calling the private methods in this file.

    Parameters
    ----------
    data: pandas dataframe

    Returns
    -------
    None

    Example
    --------
    >>> run_checks(dataset)
    """
    print('Beginning testing...')
    features, predictions, true_labels = __data_preprocessing(data)
    __check_numerical_x_y(features, predictions, true_labels)
    __check_binary_class(predictions, true_labels)
    print('No errors, finished testing.')
    return