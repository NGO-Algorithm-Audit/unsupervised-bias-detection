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
        The target label (true label) for the diabetes hospital dataset is readmit_30_days.
        One could use the variable readmit_binary as the target instead.

    See Also
    --------
    Dataset details: https://fairlearn.org/main/user_guide/datasets/diabetes_hospital_data.html

    Example
    --------
    >>> from unsupervised_bias_detection.utils import load_default_dataset
    >>> x, y = load_default_dataset()
    >>> x
    pandas.dataframe(  race  gender                    age  ... had_outpatient_days readmitted  readmit_binary
    0             Caucasian  Female  '30 years or younger'  ...               False         NO               0
    1             Caucasian  Female  '30 years or younger'  ...               False        >30               1
    2       AfricanAmerican  Female  '30 years or younger'  ...                True         NO               0
    3             Caucasian    Male          '30-60 years'  ...               False         NO               0
    4             Caucasian    Male          '30-60 years'  ...               False         NO               0
    ...                 ...     ...                    ...  ...                 ...        ...             ...
    101761  AfricanAmerican    Male        'Over 60 years'  ...               False        >30               1
    101762  AfricanAmerican  Female        'Over 60 years'  ...               False         NO               0
    101763        Caucasian    Male        'Over 60 years'  ...                True         NO               0
    101764        Caucasian  Female        'Over 60 years'  ...               False         NO               0
    101765        Caucasian    Male        'Over 60 years'  ...               False         NO               0)

    [101766 rows x 24 columns]
    >>> y
    pandas.series(  0         0
                    1         0
                    2         0
                    3         0
                    4         0
                             ..
                    101761    0
                    101762    0
                    101763    0
                    101764    0
                    101765    0)
    Name: readmit_30_days, Length: 101766, dtype: int64
    """
    print('Note: it is up to the user to train a model with the provided data now before running the bias detection '
          'tool whether it is via the Algorithm Audit website for a demo or via the unsupervised_bias_detection '
          'package.')
    diabetes_dataset_x, diabetes_dataset_y = data.fetch_diabetes_hospital(return_X_y=True)
    return diabetes_dataset_x, diabetes_dataset_y