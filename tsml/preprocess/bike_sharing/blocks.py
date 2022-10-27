import pandas as pd

from tsml.base import ExtractPandasCSVMixin
from tsml.base import ExportPandasParquetMixin
from tsml.preprocess.helpers import roll, lag
from tsml.utils.logger import log_execution


class BikeSharingProcessBlock(ExtractPandasCSVMixin, ExportPandasParquetMixin):

    name = 'Bike sharing process block'

    def __init__(self, import_path:str, export_path:str, import_kwargs:dict):
        self.import_path   = import_path
        self.export_path   = export_path
        self.import_kwargs = import_kwargs
    
    @log_execution
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df
            .pipe(roll, column='count', roll=7, aggregation='mean')
            .pipe(roll, column='count', roll=30, aggregation='mean')
            .pipe(lag,  column='count', periods=1)
            .dropna()
        )

    @log_execution(name=name)
    def process(self):
        df = self.extract(**self.import_kwargs)
        df = self.transform(df)
        self.export(df)
