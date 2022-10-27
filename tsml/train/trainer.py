import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate

from tsml.base import ExportPickleMixin

class CustomTimeSeriesSplit(TimeSeriesSplit):
    """
    Extend TimeSeriesSplit functionality to yield the original unique indexes
    This is convinient when many products are contained in the same dataset, and not just
    one time series
    """
    def split(self, X, date_column: str, y=None, groups=None):
        unique_dates = X[date_column].unique()
        for train_dates, test_dates in super().split(unique_dates, y, groups):
            train_index = X.index[X[date_column].isin(unique_dates[train_dates])]
            test_index  = X.index[X[date_column].isin(unique_dates[test_dates])]
            yield (train_index.to_numpy(), test_index.to_numpy())

class TSTrainer(ExportPickleMixin):
    def __init__(self, train_params: dict) -> None:
        self.train_params = train_params
        self.cv_scores = None

    def validate(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.cv_scores = cross_validate(
            pipeline, 
            X_train,
            y_train,
            cv = TimeSeriesSplit(**self.train_params['cv']).split(X_train), 
            # cv = CustomTimeSeriesSplit(**self.train_params['cv']).split(X_train, date_column='date'), 
            scoring = self.train_params['scoring'],
            verbose = self.train_params['verbose']
        )

    def test(
        self,
        pipeline: Pipeline, 
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame,
        X_test: pd.DataFrame, 
        y_test: pd.DataFrame,
    ):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        self.test_scores = { 
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred)
        }

