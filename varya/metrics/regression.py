import numpy
import pandas
import torch

from ..utils.validation import (
    recast_to_tensor,
    verify_consistent_rows,
    probe_regression_targets,
)

def max_error(y_true, y_pred):
    """
    The max_error metric calculates the maximum residual error.

    Parameters
    ----------
    y_true : list / np.array / torch.Tensor of shape (num_rows,)
        Ground truth (correct) target values.
    y_pred : list / np.array / torch.Tensor of shape (num_rows,)
        Estimated target values.
    
    Returns
    -------
    max_error : float
        A positive floating point value (the best value is 0.0).
    
    Examples
    --------
    >>> from varya.metrics import max_error
    >>> y_true = [3, 2, 7, 1]
    >>> y_pred = [4, 2, 7, 1]
    >>> max_error(y_true, y_pred)
    1
    """
    y_true = recast_to_tensor(y_true)
    y_pred = recast_to_tensor(y_pred)

    verify_consistent_rows(y_true, y_pred)

    y_true, y_pred = probe_regression_targets(y_true, y_pred)

    return torch.absolute(y_true - y_pred).max().item()

def mean_absolute_error(y_true, y_pred):

    y_true = recast_to_tensor(y_true, dtype=torch.float64)
    y_pred = recast_to_tensor(y_pred, dtype=torch.float64)

    verify_consistent_rows(y_true, y_pred)

    y_true, y_pred = probe_regression_targets(y_true, y_pred)

    return torch.absolute(y_true - y_pred).mean().item()

def mean_squared_error(y_true, y_pred):

    y_true = recast_to_tensor(y_true, dtype=torch.float64)
    y_pred = recast_to_tensor(y_pred, dtype=torch.float64)

    verify_consistent_rows(y_true, y_pred)

    y_true, y_pred = probe_regression_targets(y_true, y_pred)

    return torch.pow(y_true - y_pred, 2).mean().item()

def root_mean_squared_error(y_true, y_pred):

    y_true = recast_to_tensor(y_true, dtype=torch.float64)
    y_pred = recast_to_tensor(y_pred, dtype=torch.float64)

    verify_consistent_rows(y_true, y_pred)

    y_true, y_pred = probe_regression_targets(y_true, y_pred)

    return torch.pow(torch.pow(y_true - y_pred, 2).mean(), 0.5).item()