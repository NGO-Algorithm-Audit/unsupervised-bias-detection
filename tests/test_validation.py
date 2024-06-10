"""Provides tests for the functions in unsupervised_bias_detection/utils/validation.py."""
import pandas as pd
import numpy as np
import pytest

from unsupervised_bias_detection.utils.validation import run_checks


def test_always_passes():
    """Test0: all numerical and good (no errors expected)."""
    dict0 = {'x': [[1, 2, 3], [3, 2, 1],[4, 5, 6]], 'preds': [0, 1, 1], 'true_labels': [0, 0, 1]}
    df_test0 = pd.DataFrame(data=dict0)
    assert not run_checks(df_test0) is ValueError

@pytest.mark.xfail
def test_not_binary_y():
    """Test1: all numerical BUT predictions and labels are not binary."""
    dict1 = {'x': [[1, 2, 3], [3, 2, 1],[4, 5, 6]], 'preds': [6, 7, 8], 'true_labels': [11, 0, 2]}
    df_test1 = pd.DataFrame(data=dict1)
    assert run_checks(df_test1) is ValueError

@pytest.mark.xfail
def test_categorical_preds():
    """Test2: all numerical BUT predictions are categorical."""
    dict2 = {'x': [[1, 2, 3], [3, 2, 1],[4, 5, 6]], 'preds': ['yellow', 'yellow', 'blue'], 'true_labels': [0, 1, 1]}
    df_test2 = pd.DataFrame(data=dict2)
    assert run_checks(df_test2) is ValueError

@pytest.mark.xfail
def test_categorical_true_labels():
    """Test3: all numerical BUT true labels are categorical."""
    dict3 = {'x': [[1, 2, 3], [3, 2, 1],[4, 5, 6]], 'preds': [0, 1, 0], 'true_labels':  ['red', 'red', 'yellow']}
    df_test3 = pd.DataFrame(data=dict3)
    assert run_checks(df_test3) is ValueError

@pytest.mark.xfail
def test_multiclass_preds():
    """Test4: all numerical BUT predictions are multi-class."""
    dict4 = {'x': [[1, 2, 3], [3, 2, 1],[4, 5, 6]], 'preds': [0,1,2], 'true_labels': [0, 1, 1]}
    df_test4 = pd.DataFrame(data=dict4)
    assert run_checks(df_test4) is ValueError

@pytest.mark.xfail
def test_multiclass_true_labels():
    """Test5: all numerical BUT true labels are multi-class."""
    dict5 = {'x': [[1, 2, 3], [3, 2, 1],[4, 5, 6]], 'preds': [0, 1, 1], 'true_labels': [0,1,2]}
    df_test5 = pd.DataFrame(data=dict5)
    assert run_checks(df_test5) is ValueError

@pytest.mark.xfail
def test_features_nonnumerical():
    """Test6: x includes categorical values."""
    dict6 = {'x': [[1, 'three', 2], ['blue', 100, 0], [0, 0, 0]], 'preds': [0, 1, 1], 'true_labels': [1, 1, 1]}
    df_test6 = pd.DataFrame(data=dict6)
    assert run_checks(df_test6) is ValueError

@pytest.mark.xfail
def test_two_missing_columns():
    """Test7: only features present, missing predictions and true labels."""
    dict7 = {'x': [[1, 2, 3], [3, 2, 1],[4, 5, 6]]}
    df_test7 = pd.DataFrame(data=dict7)
    assert run_checks(df_test7) is IndexError

@pytest.mark.xfail
def test_missing_true_labels():
    """Test8: true labels column missing."""
    dict8 = {'x': [[1, 'three', 2], ['blue', 100, 0], [0, 0, 0]], 'preds': [0, 1, 1]}
    df_test8 = pd.DataFrame(data=dict8)
    assert run_checks(df_test8) is IndexError

@pytest.mark.xfail
def test_missing_features():
    """Test9: features missing."""
    dict9 = {'preds': [0, 1, 1], 'true_labels': [0, 1, 1]}
    df_test9 = pd.DataFrame(data=dict9)
    assert run_checks(df_test9) is IndexError

@pytest.mark.xfail
def test_not_pandas_type():
    """Test10: the data is not of type pandas."""
    array10 = np.array([[1, 2, 3, 0, 1], [4, 5, 6, 0, 0], [7, 8, 9, 1, 1]])
    assert run_checks(array10) is TypeError