import numpy as np
import pandas as pd
from tsml.base import ExtractPandasParquetMixin

class DataLoader(ExtractPandasParquetMixin):
    def __init__(
        self, import_path: str, 
        target: str, 
        window_split: dict, 
        split_target:bool = True, 
        index:str = 'date'
    ):
        self.import_path = import_path
        self.target = target
        self.window_split = window_split
        self.index = index
        self.split_target = split_target

        self._X = None
        self._X_train = None
        self._y_train = None
        self._y_test = None
        self._X_test = None
        self._df = None

    @property
    def df(self):
        if self._df is None:
            self._df = self.extract()
            if self.index:
                self._df = self._df.set_index(self.index, drop=False)

        return self._df

    @property
    def X(self):
        if self.split_target:
            return self.df[[col for col in self.df.columns if col not in self.target]]
        else:
            return self.df
    
    @property
    def y(self):
        return self.df[self.target]

    @property
    def X_train(self):
        if self._X_train is None:
            self.split()
        return self._X_train

    @property
    def y_train(self):
        if self._y_train is None:
            self.split()
        return self._y_train

    @property
    def X_test(self):
        if self._X_test is None:
            self.split()
        return self._X_test

    @property
    def y_test(self):
        if self._y_test is None:
            self.split()
        return self._y_test

    def split(self): ## test_size equal to the time horizon. Splits defines the rest
        if isinstance(self.X.index, pd.core.indexes.range.RangeIndex):
            idx_u = np.unique(self.X.index.values)
        else: 
            idx_u = self.X.index.unique()

        train_index = idx_u[ : -self.window_split]
        test_index  = idx_u[ -self.window_split : ]
        self._X_train, self._X_test = self.X.loc[train_index], self.X.loc[test_index]
        self._y_train, self._y_test = self.y.loc[train_index], self.y.loc[test_index]
