import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import SplineTransformer, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

##########################################################################################
######## Bike sharing pipelines

## Columns
categorical_columns = [
    'weather',
    'season',
    'holiday',
    'workingday',
]

ordinal_encoder = OrdinalEncoder()
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation='periodic',
        include_bias=True,
    )

## Transformers
cyclic_spline_transformer = ColumnTransformer(
    transformers=[
        ('categorical', one_hot_encoder, categorical_columns),
        ('cyclic_month', periodic_spline_transformer(12, n_splines=6), ['month']),
        ('cyclic_weekday', periodic_spline_transformer(7, n_splines=3), ['weekday']),
        ('cyclic_hour', periodic_spline_transformer(24, n_splines=12), ['hour']),
    ],
    remainder=MinMaxScaler(),
)

## Pipelines
xgb_pipeline_1 = make_pipeline(
    ColumnTransformer(
        transformers=[
            ('categorical', ordinal_encoder, categorical_columns),
        ],
        remainder='passthrough',
    ),
    XGBRegressor(),
)
lgbm_pipeline_1 = make_pipeline(
    ColumnTransformer(
        transformers=[
            ('categorical', ordinal_encoder, categorical_columns),
        ],
        remainder='passthrough',
    ),
    LGBMRegressor(),
)
xgb_pipeline_2 = make_pipeline(
    ColumnTransformer(
        transformers=[
            ('categorical', one_hot_encoder, categorical_columns),
        ],
        remainder=MinMaxScaler(),
    ),
    XGBRegressor(),
)
xgb_pipeline_3 = make_pipeline(cyclic_spline_transformer, XGBRegressor())

