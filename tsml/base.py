import dask.dataframe as dd
import joblib
import pandas as pd
from tsml.utils.logger import log_execution, logger

#### Export
class ExportPickleMixin:
    @log_execution
    def export(self, x):
        logger.info(f'Exporting {self.export_path}')
        joblib.dump(x, self.export_path)

class ExportPandasPickleMixin:
    @log_execution
    def export(self, df: pd.DataFrame) -> None:
        logger.info(f'Exporting {self.export_path}')
        df.to_pickle(self.export_path)

class ExportPandasParquetMixin:
    @log_execution
    def export(self, df: pd.DataFrame) -> None:
        logger.info(f'Exporting {self.export_path}')
        df.to_parquet(self.export_path)

class ExportDaskParquetMixin:
    @log_execution
    def export(self, df) -> None:
        logger.info(f'Exporting {self.export_path}')
        dd.to_parquet(self.export_path)

#### Extract
class ExtractPandasParquetMixin:
    @log_execution
    def extract(self):
        logger.info(f'Loading {self.import_path}')
        if isinstance(self.import_path, list):
            dfs = [pd.read_parquet(path) for path in self.import_path]
            return dfs
        else:
            return pd.read_parquet(self.import_path)

class ExtractDaskParquetMixin:
    @log_execution
    def extract(self):
        logger.info(f'Loading {self.import_path}')
        if isinstance(self.import_path, list):
            dfs = [dd.read_parquet(path) for path in self.import_path]
            return dfs
        else:
            return dd.read_parquet(self.import_path)

class ExtractPandasCSVMixin:
    @log_execution
    def extract(self, **kwargs):
        logger.info(f'Loading {self.import_path}')
        if isinstance(self.import_path, list):
            dfs = [pd.read_csv(path, **kwargs) for path in self.import_path]
            return dfs
        else:
            return pd.read_csv(self.import_path, **kwargs)

class ExtractDaskCSVMixin:
    @log_execution
    def extract(self, **kwargs):
        logger.info(f'Loading {self.import_path}')
        if isinstance(self.import_path, list):
            dfs = [dd.read_csv(path, **kwargs) for path in self.import_path]
            return dfs
        else:
            return dd.read_csv(self.import_path, **kwargs)