import typing as tp
from enum import Enum, auto
import pandas as pd
from etna.transforms.base import Transform
from etna.pipeline import Pipeline
from etna.transforms import \
(
    TimeSeriesImputerTransform,
    MedianOutliersTransform,
    STLTransform,
    TrendTransform,
    ChangePointsSegmentationTransform,
    TrendTransform,
    SegmentEncoderTransform,
    MeanEncoderTransform,
    MeanSegmentEncoderTransform,
    LabelEncoderTransform,
    TreeFeatureSelectionTransform,
    DateFlagsTransform,
    HolidayTransform,
    LagTransform,
    ExogShiftTransform,
    MeanTransform,
    SumTransform,
    MedianTransform,
    MaxTransform,
    MinTransform,
    QuantileTransform,
    StdTransform,
    MADTransform,
    MinMaxDifferenceTransform,
    StandardScalerTransform,
    RobustScalerTransform,
    MinMaxScalerTransform,
    MaxAbsScalerTransform,
    LogTransform,
    YeoJohnsonTransform,
    BoxCoxTransform,
    DifferencingTransform
)
from etna.models import \
(
    MovingAverageModel,
    SARIMAXModel,
    HoltWintersModel,
    SimpleExpSmoothingModel,
    CatBoostPerSegmentModel,
    AutoARIMAModel
)
from etna.metrics import \
(
    RMSE,
    MAE,
    SMAPE
)

from etna.datasets import TSDataset

class ModelType(Enum):
    MA = "MA"
    ARIMA = "ARIMA"
    SARIMAX = "SARIMAX"
    PROPHET = "PROPHET"
    CB = "CB"

class Forecaster:
    def generate_features(self):
        return [
            TimeSeriesImputerTransform(in_column="sell_price", strategy="seasonal_nonautoreg", seasonality=self.horizon),
            STLTransform(in_column='target', period=7),
            # TrendTransform(in_column="target"),
            # TrendTransform(in_column="sell_price"),
            # ChangePointsSegmentationTransform(in_column="target"),
            # ChangePointsSegmentationTransform(in_column="sell_price"),
            # SegmentEncoderTransform(),
            # MeanEncoderTransform(in_column="target"),
            # MeanEncoderTransform(in_column="sell_price"),
            # DateFlagsTransform(
            #     in_column="timestamp",
            #     day_number_in_week=True,
            #     day_number_in_month=True,
            #     day_number_in_year=True,
            #     week_number_in_month=True,
            #     week_number_in_year=True,
            #     month_number_in_year=True,
            #     season_number=True,
            #     year_number=True,
            #     is_weekend=True
            #     ),
            # HolidayTransform(mode="binary", iso_code="USA"),
            # LagTransform(in_column="target", lags=[1, 7, 14, 30, 90, 180, 365]),
            # LagTransform(in_column="sell_price", lags=[1, 7, 14, 30, 90, 180, 365]),
            # ExogShiftTransform(horizon=self.horizon, lag="auto"),
            # MeanTransform(in_column="target", window=self.horizon),
            # MeanTransform(in_column="sell_price", window=self.horizon),
            # SumTransform(in_column="target", window=self.horizon),
            # SumTransform(in_column="sell_price", window=self.horizon),
            # MedianTransform(in_column="target", window=self.horizon),
            # MedianTransform(in_column="sell_price", window=self.horizon),
            # MaxTransform(in_column="target", window=self.horizon),
            # MaxTransform(in_column="sell_price", window=self.horizon),
            # MinTransform(in_column="target", window=self.horizon),
            # MinTransform(in_column="sell_price", window=self.horizon),
            # StdTransform(in_column="target", window=self.horizon),
            # StdTransform(in_column="sell_price", window=self.horizon),
            # MADTransform(in_column="target", window=self.horizon),
            # MADTransform(in_column="sell_price", window=self.horizon),
            # MinMaxDifferenceTransform(in_column="target", window=self.horizon),
            # MinMaxDifferenceTransform(in_column="sell_price", window=self.horizon),
            # StandardScalerTransform(in_column="target"),
            # StandardScalerTransform(in_column="sell_price"),
            # RobustScalerTransform(in_column="target"),
            # RobustScalerTransform(in_column="sell_price"),
            # MinMaxScalerTransform(in_column="target"),
            # MinMaxScalerTransform(in_column="sell_price"),
            # MaxAbsScalerTransform(in_column="target"),
            # MaxAbsScalerTransform(in_column="sell_price"),
            # LogTransform(in_column="target"),
            # LogTransform(in_column="sell_price"),
            # DifferencingTransform(in_column="target", order=1),
            # DifferencingTransform(in_column="sell_price", order=1),
            # DifferencingTransform(in_column="target", order=2),
            # DifferencingTransform(in_column="sell_price", order=2),
            # QuantileTransform(in_column="target", window=self.horizon, quantile=0.25),
            # QuantileTransform(in_column="target", window=self.horizon, quantile=0.50),
            # QuantileTransform(in_column="target", window=self.horizon, quantile=0.75),
            # QuantileTransform(in_column="target", window=self.horizon, quantile=0.90),
            # QuantileTransform(in_column="target", window=self.horizon, quantile=0.95),
            # QuantileTransform(in_column="sell_price", window=self.horizon, quantile=0.25),
            # QuantileTransform(in_column="sell_price", window=self.horizon, quantile=0.50),
            # QuantileTransform(in_column="sell_price", window=self.horizon, quantile=0.75),
            # QuantileTransform(in_column="sell_price", window=self.horizon, quantile=0.90),
            # QuantileTransform(in_column="sell_price", window=self.horizon, quantile=0.95),
        ]
    
    def __init__(self, model_name: str, horizon: int, sales_data: pd.DataFrame, prices_data: pd.DataFrame, calendar_data: pd.DataFrame, store_id: str):
        self.horizon = horizon
        self.sales_data = sales_data
        self.prices_data = prices_data
        self.calendar_data = calendar_data
        self.store_id = store_id
        
        transformations = [
            MedianOutliersTransform(in_column="target", window_size=horizon),
            TimeSeriesImputerTransform(strategy="seasonal_nonautoreg", seasonality=horizon),
        ]

        match model_name:
            case ModelType.MA.value:
                self.model = MovingAverageModel(window=horizon)

            case ModelType.ARIMA.value:
                self.model = AutoARIMAModel(order=(1, 0, 0), seasonal_order=(1, 1, 1, 7))
                transformations.extend(
                    [
                        MeanTransform(in_column='target', window=7, alpha=0.9),
                        DifferencingTransform(in_column="target", order=1)
                    ]
                )

            case ModelType.SARIMAX.value:
                self.model = SARIMAXModel(seasonal_order=(1, 1, 1, 7))
                transformations.extend(
                    [
                        MeanTransform(in_column='target', window=7, alpha=0.9),
                        DifferencingTransform(in_column="target", order=1)
                    ]
                )
                transformations.extend(
                        self.generate_features()
                )

            case _:
                raise ValueError(f"Unrecognized model `{model_name}`")
        
        self.transformations = transformations
        self.pipeline = Pipeline(model=self.model, transforms=self.transformations, horizon=self.horizon)
        self.metrics = [RMSE, MAE, SMAPE]
    
    def preprocess_data(self):
        store_sales = self.sales_data[self.sales_data["store_id"] == self.store_id]
        store_prices = self.prices_data[self.prices_data["store_id"] == self.store_id]

        data = store_sales \
        .merge(self.calendar_data, on="date_id", how="left", suffixes=("", "x")) \
        .merge(store_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", suffixes=("", "y"))

        data.sort_values(by="date_id", inplace=True)

        drop_columns = ["date_id", "wm_yr_wk", "store_id", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "weekday"]
        data.drop(columns=drop_columns, inplace=True)

        data.rename(columns={"item_id": "segment", "date": "timestamp", "cnt": "target"}, inplace=True)

        df_main = TSDataset.to_dataset(data[["timestamp", "segment", "target"]])
        df_exog = TSDataset.to_dataset(data.drop(columns=["target"]))
        ts_data = TSDataset(df=df_main, freq="D", df_exog=df_exog)

        train_data, test_data = ts_data.train_test_split(test_size=self.horizon)

        self.train_data = train_data
        self.test_data = test_data

    def fit(self):
        self.pipeline.fit(self.train_data)
    
    def predict(self):
        return self.pipeline.forecast()
    
    def eval(self):

        metrics_dict = {}
        predictions = self.predict()
        for metric in self.metrics:
            metrics_dict[metric.__name__] = metric()(y_true=self.test_data, y_pred=predictions)
        
        metrics_df = pd.DataFrame.from_records(
            data=[metrics_dict[metric_name] for metric_name in metrics_dict],
            index=[metric_name for metric_name in metrics_dict]
        ).T

        return metrics_df
        

