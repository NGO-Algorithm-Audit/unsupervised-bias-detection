"""Provides tests for the loading function in unsupervised_bias_detection/utils/dataset.py."""
import pytest

from unsupervised_bias_detection.utils.dataset import load_default_dataset


def test_loading_dataset_passes():
    """Checks that the default dataset loading function works as expected."""
    data, true_labels = load_default_dataset()
    assert data is not None and true_labels is not None

@pytest.mark.xfail
def test_unneeded_argument():
    """Checks that no argument is necessary for the function call."""
    assert load_default_dataset(False) is TypeError