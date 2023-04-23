import torch

from ..utils.validation import (
    recast_to_tensor
)

def entropy(y_labels):
    '''
    Calulate the entropy of given label list
    
    :param label_array: a numpy array of binary labels shape = (n, 1)
    :return entropy: entropy value
    '''
    y_labels = recast_to_tensor(y_labels, dtype=torch.float64)

    labels, counts = torch.unique(y_labels, return_counts=True)
    pm = counts / len(y_labels)

    entropy = 0
    for pm_i in pm:
        entropy -= (pm_i * np.log(pm_i))
    return entropy

def gini_index(y_labels):
    '''
    Calulate the gini index of label list
    
    :param label_array: a numpy array of labels shape = (n, 1)
    :return gini: gini index value
    '''
    y_labels = recast_to_tensor(y_labels, dtype=torch.float64)

    labels, counts = torch.unique(y_labels, return_counts=True)
    pm = counts / len(y_labels)

    gini = 0
    for pm_i in pm:
        gini += pm_i * (1 - pm_i)
    return gini