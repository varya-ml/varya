import pandas
import numpy
import torch

class Datum:

    def __init__(self, data, target=None):
        
        self._data = None
        self._target = None
        self._column_names = None
        self._target_name = None

        if target:
            self.data = data
            self.target = target
        else:
            self.data = data

    @property
    def data(self):

        return self.recast(self._data)

    @data.setter
    def data(self, data):

        self._data = data
        if isinstance(data, (pandas.core.frame.DataFrame, pandas.core.series.Series)):
            self._column_names = data.columns.tolist()

    @property
    def target(self):

        if self._target is not None:
            return self.recast(self._target).view(-1, 1)
        else:
            raise Exception('target not defined.')

    @target.setter
    def target(self, target):

        if isinstance(target, str):
            self._target = self._data[target]
            self._data = self._data.drop([target], axis=1)
            self._target_name = target            
            self._column_names = self._data.columns.tolist()

        elif isinstance(target, int):
            self._column_names = None
            self._target_name = None
            self._data = self.recast(self._data)
            self._target = self._data[:, target]
            self._data = self._data[:, np.arange(self._data.shape[1]) != target]
        
        elif isinstance(target, (pandas.core.frame.DataFrame, pandas.core.series.Series,
                                 numpy.ndarray,
                                 torch.Tensor)):
            self._target = target
            if isinstance(target, (pandas.core.frame.DataFrame, pandas.core.series.Series)):
                self._target_name = target.name

    def recast(self, datum):

        if isinstance(datum, (pandas.core.frame.DataFrame, pandas.core.series.Series)):
            return torch.from_numpy(datum.values)
        
        if isinstance(datum, numpy.ndarray):
            return torch.from_numpy(datum)

        if isinstance(datum, torch.Tensor):
            return datum