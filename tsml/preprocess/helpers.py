import numpy as np
import pandas as pd


def roll(df: pd.DataFrame, column:str, roll:int, aggregation:str = 'mean') -> pd.DataFrame:
    if aggregation == 'mean':
        df[column + f'_mean_roll_{roll}'] = df[column].rolling(roll).mean()
    if aggregation == 'sum':
        df[column + f'_sum_roll_{roll}'] = df[column].rolling(roll).sum()
    return df

def grouped_roll(df: pd.DataFrame, group:list, column:str, roll:int, aggregation:str = 'mean') -> pd.DataFrame:
    grouped_df = df.groupby(group)
    if aggregation == 'mean':
        df[column + f'_mean_roll_{roll}'] = grouped_df[column].transform(lambda s: s.rolling(roll).mean())
    if aggregation == 'sum':
        df[column + f'_sum_roll_{roll}'] = grouped_df[column].transform(lambda s: s.rolling(roll).sum())
    return df

def lag(df: pd.DataFrame, column:str, periods:int) -> pd.DataFrame:
    df[column + f'_lag_{periods}'] = df[column].shift(periods=periods)
    return df

def grouped_lag(df: pd.DataFrame, group:list, column:str, periods:list) -> pd.DataFrame:
    grouped_df = df.groupby(group)
    df[column + f'_lag_{periods}'] = grouped_df[column].shift(periods)
    return df

def censored_data(df:pd.DataFrame, column:str):
    df['is_censored'] = (df[column] == 0)*1
    return df

def grouped_shift_target(df: pd.DataFrame, group:list, column:str, shifts:list) -> pd.DataFrame:
    grouped_df = df.groupby(group)
    for s in shifts:
        df[f'y_{s}'] = grouped_df[column].shift(-s)
    return df

def total_grouped_target(df: pd.DataFrame, group:list, target:str):
    df['id_y_0'] = df.groupby(group)[target].transform(sum)
    return df

def parse_date(df: pd.DataFrame, column:str, format='%Y%m%d') -> pd.DataFrame:
    df[column] = pd.to_datetime(df[column], format=format)
    return df

def mean_price(df: pd.DataFrame, group_cols: list, price_column:str):
    df['mean_price'] = df.groupby(group_cols)[price_column].transform(np.mean)
    return df

def replace_nan_price(df: pd.DataFrame, replace_col: str, replace_value: str) -> pd.DataFrame:
    df[replace_col] = df[replace_col].fillna(df[replace_value])
    return df