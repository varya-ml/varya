import warnings
from math import ceil, floor

import numpy
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from ..compose.core import (
    Datum
)

def _validate_shuffle_split(n_rows, test_size, train_size, default_test_size=None):
    """
    Validation helper to check if the test/test sizes are meaningful w.r.t. the
    size of the data (n_rows).
    """

    if test_size is None and train_size is None:
        test_size = default_test_size

    test_size_type = numpy.asarray(test_size).dtype.kind
    train_size_type = numpy.asarray(train_size).dtype.kind

    if (
        test_size_type == 'i'
        and (test_size >= n_rows or test_size <= 0)
        or test_size_type == 'f'
        and (test_size <= 0 or test_size >= 1)
    ):
        raise ValueError(
            f'''test_size={test_size} should be either positive and smaller 
            than the number of samples {n_rows} or a float in the 
            (0, 1) range'''
        )

    if (
        train_size_type == 'i'
        and (train_size >= n_rows or train_size <= 0)
        or train_size_type == 'f'
        and (train_size <= 0 or train_size >= 1)
    ):
        raise ValueError(
            f'''test_size={train_size} should be either positive and smaller 
            than the number of samples {n_rows} or a float in the 
            (0, 1) range'''
        )

    if train_size is not None and train_size_type not in ('i', 'f'):
        raise ValueError(f'Invalid value for train_size: {train_size}')
    if test_size is not None and test_size_type not in ('i', 'f'):
        raise ValueError(f'Invalid value for test_size: {test_size}')

    if train_size_type == 'f' and test_size_type == 'f' and train_size + test_size > 1:
        raise ValueError(
            f'''The sum of test_size and train_size = {train_size + test_size},
            should be in the (0, 1) range. Reduce test_size and/or train_size.'''
        )

    if test_size_type == 'f':
        n_test = ceil(test_size * n_rows)
    elif test_size_type == 'i':
        n_test = float(test_size)

    if train_size_type == 'f':
        n_train = floor(train_size * n_rows)
    elif train_size_type == 'i':
        n_train = float(train_size)

    if train_size is None:
        n_train = n_rows - n_test
    elif test_size is None:
        n_test = n_rows - n_train

    if n_train + n_test > n_rows:
        raise ValueError(
            '''The sum of train_size and test_size = %d, 
            should be smaller than the number of 
            samples %d. Reduce test_size and/or 
            train_size.''' % (n_train + n_test, n_rows)
        )

    n_train, n_test = int(n_train), int(n_test)

    if n_train == 0:
        raise ValueError(
            f'''With n_samples={n_rows}, test_size={test_size} and train_size={train_size}, 
            the resulting train set will be empty. Adjust any of the aforementioned parameters.'''
        )

    return n_train, n_test

def train_test_split(X, y=None, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    """Split tensors or datum into random train and test subsets.
    Quick utility that wraps input validation,
    Parameters
    ----------
    X : tensor array or datum object
    y : tensor array or datum object
    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.
        Read more in the :ref:`User Guide <stratification>`.
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
        .. versionadded:: 0.16
            If the input is sparse, the output will be a
            ``scipy.sparse.csr_matrix``. Else, output type is the same as the
            input type.
    Examples
    --------
    >>> import numpy as np
    >>> from varya.model_selection import train_test_split
    >>> data = Datum(df, 0)
    >>> data.X
    tensor([[0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]])
    >>> list(data.y)
    [0, 1, 2, 3, 4]
    >>> train, test = train_test_split(
    ...     data, test_size=0.33, random_state=42)
    ...
    >>> train.X
    array([[4, 5],
           [0, 1],
           [6, 7]])
    >>> train.y
    [2, 0, 3]
    >>> test.X
    array([[2, 3],
           [8, 9]])
    >>> test.y
    [1, 4]
    """

    if hasattr(X, '_data') and hasattr(X, '_target') and y is not None:
        warnings.warn('Any target present in Datum object passed will be replaced with new y')
        data = X
        data.y = y
    elif hasattr(X, '_data') and hasattr(X, '_target') and not y:
        data = X
    else:
        data = Datum(X, y)
    
    n_train, n_test = _validate_shuffle_split(data.rows, test_size, train_size, default_test_size=0.25)

    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                'Stratified train/test split is not implemented for shuffle=False'
            )

        train_indices = numpy.arange(n_train)
        test_indices = numpy.arange(n_train, n_train + n_test)

    else:
        if stratify is not None:
            CVClass = StratifiedShuffleSplit
        else:
            CVClass = ShuffleSplit

        cv = CVClass(test_size=n_test, train_size=n_train, random_state=random_state)

        train_indices, test_indices = next(cv.split(X=data.X, y=stratify))

    train, test = Datum(data.X[train_indices], data.y[train_indices]), Datum(data.X[test_indices], data.y[test_indices])
    return train, test