import pandas as pd

from tsml.base import ExtractPandasCSVMixin, ExtractPandasParquetMixin 
from tsml.base import ExportPandasParquetMixin
from tsml.preprocess.helpers import (
    mean_price, parse_date, replace_nan_price, grouped_shift_target, 
    total_grouped_target, grouped_lag, grouped_roll, censored_data
)
from tsml.utils.logger import log_execution
from tsml.settings import WALMART_FORECAST_HORIZON

WALMART_FORECAST_HORIZON = 14

class WalmartProcessBlockZero(ExtractPandasCSVMixin, ExportPandasParquetMixin):

    name = 'Walmart process block zero'

    def __init__(self, import_path:list, export_path:str):
        self.import_path   = import_path
        self.export_path   = export_path

    @log_execution
    def transform(
        self, 
        sales_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        return (sales_df
            .melt(
                id_vars=[col for col in sales_df.columns if not 'd_' in col],
                value_name='y_0'
            )
            .drop(columns='id')
            .merge(
                calendar_df
                    .fillna('None')
                    .pipe(parse_date, 'date', format='%Y-%m-%d'),
                left_on='variable', 
                right_on='d', 
                how='left'
            )
            .drop(columns=['d','variable','event_type_1','event_type_2'])
            .merge(
                prices_df, 
                on=['store_id','item_id','wm_yr_wk'],
                how='left'
            )
            .drop(columns=['wm_yr_wk'])
            .pipe(mean_price, ['item_id','store_id'], 'sell_price')
            .pipe(replace_nan_price, 'sell_price', 'mean_price')
            .drop(columns=['mean_price','weekday', 'wday', 'month', 'year',])
            .reset_index(drop=True)
        )    

    @log_execution(name=name)
    def process(self):
        sales_df, calendar_df, prices_df = self.extract()
        total_sales = self.transform(
            sales_df    = sales_df,
            calendar_df = calendar_df,
            prices_df   = prices_df
        )
        self.export(total_sales)


class WalmartProcessBlockOne(ExtractPandasParquetMixin, ExportPandasParquetMixin):

    name = 'Walmart process block one'

    def __init__(self, import_path:str, export_path:str, group_by: list, drop_columns:list):
        self.import_path   = import_path
        self.export_path   = export_path
        self.group_by      = group_by
        self.drop_columns  = drop_columns

    @log_execution
    def transform(self, df: pd.DataFrame):
        shift_group = [col for col in self.group_by if 'date' not in col]
        shifts = [n for n in range(1, WALMART_FORECAST_HORIZON)]
        return (df
            .pipe(total_grouped_target, self.group_by, 'y_0')
            .drop(columns=['y_0'])
            .rename(columns={'id_y_0':'y_0'})
            .drop(columns=self.drop_columns)
            .groupby(self.group_by).nth(0)
            .reset_index()
            .pipe(grouped_shift_target, shift_group, 'y_0', shifts) 
            .pipe(grouped_roll, group=shift_group, column='y_0', roll=7, aggregation='mean')
            .pipe(grouped_roll, group=shift_group, column='y_0', roll=30, aggregation='mean')
            .pipe(grouped_lag,  group=shift_group, column='y_0', periods=1)
            .pipe(grouped_lag,  group=shift_group, column='y_0', periods=7)
            .pipe(censored_data, column='y_0_mean_roll_7')
            .fillna(value={f'y_{n}':-1 for n in range(1, WALMART_FORECAST_HORIZON)})
        )

    @log_execution(name=name)
    def process(self):
        df = self.extract()
        processed_df = self.transform(df)
        self.export(processed_df)
        