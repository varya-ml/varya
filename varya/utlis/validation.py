import pandas
import numpy
import torch

# verify - only verifies
# probe - returns if true or raises error
# fetch - returns the desired value
# recast - converts to desired type

def verify_array(data):
    return isinstance(data, numpy.ndarray)

def verify_dataframe(data):
    return isinstance(data, pandas.core.frame.DataFrame) 

def verify_series(data):
    return isinstance(data, pandas.core.series.Series)

def verify_tensor(data):
    return isinstance(data, torch.Tensor)

def verify_list(data):
    return isinstance(data, list)

def probe_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor
    else:
        raise TypeError(f'Expected torch.Tensor, got {type(tensor)}.')

def fetch_num_rows(tensor):
    if verify_tensor(tensor):
        return tensor.shape[0]

def fetch_num_columns(tensor):
    if verify_tensor(tensor):
        return tensor.shape[1]

def verify_consistent_rows(*tensors):
    """
    Check that all tensors have consistent first dimensions.
    Checks whether all objects in tensors have the same number of rows or length.

    Parameters
    ----------
    *tensors : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """
    num_rows = [fetch_num_rows(tensor) for tensor in tensors if tensor is not None]
    unique = len(set(num_rows))
    if unique > 1:
        raise ValueError(f'Found input variables with inconsistent numbers of rows: {[int(l) for l in num_rows]}')

def probe_regression_targets(y_true, y_pred):

    verify_consistent_rows()
    y_true = probe_tensor(y_true)
    y_pred = probe_tensor(y_pred)

    if y_true.ndimension() == 1:
        y_true = y_true.view(-1, 1)
    
    if y_pred.ndimension() == 1:
        y_pred = y_pred.view(-1, 1)

    if fetch_num_rows(y_true) != fetch_num_rows(y_pred):
        raise ValueError(
            f'y_true and y_pred have different number of rows ({fetch_num_rows(y_true)}!={fetch_num_rows(y_pred)})'
            )
    
    return  y_true, y_pred

def recast_to_tensor(data, dtype=torch.float64):
    
    if verify_array(data):
        return torch.from_numpy(data).to(dtype)
    
    if verify_dataframe(data):
        return torch.from_numpy(data.values).to(dtype)
    
    if verify_series(data):
        return torch.from_numpy(data.values).to(dtype)
    
    if verify_list(data):
        return torch.tensor(data).to(dtype)
    
    if verify_tensor(data):
        return data.to(dtype)