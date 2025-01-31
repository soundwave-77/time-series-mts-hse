from enum import Enum
import pandas as pd
from etna.pipeline import Pipeline
from statsmodels.tsa.stattools import adfuller, kpss
from pathlib import Path
from etna.core import load
from etna.transforms import (
    TimeSeriesImputerTransform,
    ChangePointsSegmentationTransform,
    TrendTransform,
    DateFlagsTransform,
    HolidayTransform,
    LagTransform,
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
    DifferencingTransform,
    DensityOutliersTransform,
)
from etna.models import (
    MovingAverageModel,
    SARIMAXModel,
    CatBoostPerSegmentModel,
    AutoARIMAModel,
)
from etna.metrics import RMSE, MAE, SMAPE
from etna.analysis import plot_forecast
from statsmodels.tsa.statespace.tools import diff
from etna.datasets import TSDataset
import warnings

import json

warnings.filterwarnings("ignore")


MODELS_DIR = "models"


class ModelType(Enum):
    MA = "MA"
    ARIMA = "ARIMA"
    SARIMAX = "SARIMAX"
    CATBOOST = "CATBOOST"


class Forecaster:
    def __init__(self, model_name: str, horizon: int, use_fitted: bool = False):
        self.model_name = model_name
        self.horizon = horizon
        self.use_fitted = use_fitted
        self.item_pipelines = {}
        self.predictions = {}
        self.item_data_pools = {}
        self.metrics = [RMSE, MAE, SMAPE]

    def _generate_features(self):
        """
        Функция для генерации дополнительных фич
        """
        return [
            TrendTransform(in_column="target", out_column="target_trend"),
            ChangePointsSegmentationTransform(
                in_column="target", out_column="target_change_points"
            ),
            DateFlagsTransform(
                in_column="timestamp",
                day_number_in_week=True,
                day_number_in_month=True,
                day_number_in_year=True,
                week_number_in_month=True,
                week_number_in_year=True,
                month_number_in_year=True,
                season_number=True,
                year_number=True,
                is_weekend=True,
            ),
            HolidayTransform(mode="binary", iso_code="USA", out_column="holiday"),
            MeanTransform(
                in_column="target", window=self.horizon, out_column="target_mean"
            ),
            SumTransform(
                in_column="target", window=self.horizon, out_column="target_sum"
            ),
            MedianTransform(
                in_column="target", window=self.horizon, out_column="target_median"
            ),
            MaxTransform(
                in_column="target", window=self.horizon, out_column="target_max"
            ),
            MinTransform(
                in_column="target", window=self.horizon, out_column="target_min"
            ),
            StdTransform(
                in_column="target", window=self.horizon, out_column="target_std"
            ),
            MADTransform(
                in_column="target", window=self.horizon, out_column="target_mad"
            ),
            MinMaxDifferenceTransform(
                in_column="target", window=self.horizon, out_column="target_minmax_diff"
            ),
            StandardScalerTransform(
                in_column="target", out_column="target_standard_scaled"
            ),
            RobustScalerTransform(
                in_column="target", out_column="target_robust_scaled"
            ),
            MinMaxScalerTransform(
                in_column="target", out_column="target_minmax_scaled"
            ),
            MaxAbsScalerTransform(
                in_column="target", out_column="target_maxabs_scaled"
            ),
            LogTransform(in_column="target", out_column="target_log"),
            DifferencingTransform(
                in_column="target", order=1, inplace=False, out_column="target_diff_1"
            ),
            DifferencingTransform(
                in_column="target", order=2, inplace=False, out_column="target_diff_2"
            ),
            QuantileTransform(
                in_column="target",
                window=self.horizon,
                quantile=0.25,
                out_column="target_q_0.25",
            ),
            QuantileTransform(
                in_column="target",
                window=self.horizon,
                quantile=0.50,
                out_column="target_q_0.50",
            ),
            QuantileTransform(
                in_column="target",
                window=self.horizon,
                quantile=0.75,
                out_column="target_q_0.75",
            ),
            QuantileTransform(
                in_column="target",
                window=self.horizon,
                quantile=0.90,
                out_column="target_q_0.90",
            ),
            QuantileTransform(
                in_column="target",
                window=self.horizon,
                quantile=0.95,
                out_column="target_q_0.95",
            ),
        ]

    def _test_non_stationarity(self, data):
        """
        Функция проверки ряда на нестационарность
        """

        kpsstest = kpss(data, regression="c", nlags="auto")
        dftest = adfuller(data, autolag="AIC")

        nonstationary_kpss = kpsstest[1] <= 0.05
        nonstationary_df = dftest[1] > 0.05

        return nonstationary_kpss | nonstationary_df

    def preprocess_data(
        self,
        sales_data: pd.DataFrame,
        prices_data: pd.DataFrame,
        calendar_data: pd.DataFrame,
        store_id: str,
    ):
        """
        Функция для предобработки данных
        """

        # Достаем данные по выбранному магазину
        store_sales = sales_data[sales_data["store_id"] == store_id]
        store_prices = prices_data[prices_data["store_id"] == store_id]

        # Объединяем данные из разных источников
        all_stores_data = store_sales.merge(
            calendar_data, on="date_id", how="left", suffixes=("", "x")
        ).merge(
            store_prices,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
            suffixes=("", "y"),
        )

        all_stores_data.sort_values(by="date_id", inplace=True)

        # Удаляем неиспользуемые данные
        drop_columns = [
            "date_id",
            "wm_yr_wk",
            "store_id",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "weekday",
        ]
        all_stores_data.drop(columns=drop_columns, inplace=True)

        # Переименовываем столбцы по формату etna
        all_stores_data.rename(
            columns={"item_id": "segment", "date": "timestamp", "cnt": "target"},
            inplace=True,
        )

        all_stores_data["timestamp"] = pd.to_datetime(all_stores_data["timestamp"])

        self.item_ids = all_stores_data["segment"].unique().tolist()

        for item_id in self.item_ids:
            store_data = all_stores_data[all_stores_data["segment"] == item_id]
            df_main = store_data[["timestamp", "target", "segment"]]

            df_main = TSDataset.to_dataset(df_main)
            ts_data = TSDataset(df=df_main, freq="D")

            train_data, test_data = ts_data.train_test_split(test_size=self.horizon)
            self.item_data_pools[item_id] = {"train": train_data, "test": test_data}

    def _find_diff_degree(self, data):
        """
        Функция для определения порядка дифференцирования для стационаризации ряда
        """

        degree = 0

        while self._test_non_stationarity(data):
            degree += 1
            data = diff(data, k_diff=degree)

        return degree

    def _create_pipeline_for_item(self, item_id):
        """
        Функция для создания модели конкретного товара
        """

        item_train_target = (
            self.item_data_pools[item_id]["train"][:, :, "target"]
            .values.reshape((-1,))
            .tolist()
        )

        diff_degree = 0
        if self._test_non_stationarity(item_train_target):
            diff_degree = self._find_diff_degree(item_train_target)

        transformations = [
            DensityOutliersTransform(in_column="target", distance_coef=3.0),
            TimeSeriesImputerTransform(in_column="target", strategy="running_mean"),
        ]

        match self.model_name:
            case ModelType.MA.value:
                model = MovingAverageModel(window=self.horizon)

            case ModelType.ARIMA.value:
                model = AutoARIMAModel(order=(3, diff_degree, 3))

            case ModelType.SARIMAX.value:
                model = SARIMAXModel(seasonal_order=(3, diff_degree, 3, 7))
                transformations.extend(self._generate_features())

            case ModelType.CATBOOST.value:
                model = CatBoostPerSegmentModel(iterations=200, depth=8)
                transformations.extend(self._generate_features())
                transformations.append(
                    LagTransform(
                        in_column="target",
                        lags=[1, 7, 14, 30, 90, 180, 365],
                        out_column="target_lag",
                    ),
                )
            case _:
                raise ValueError(f"Unrecognized model `{self.model_name}`")

        pipeline = Pipeline(
            model=model, transforms=transformations, horizon=self.horizon
        )

        return pipeline

    def create_pipelines(self):
        for item_id in self.item_ids:
            if self.use_fitted:
                self.load_models()
            else:
                item_pipeline = self._create_pipeline_for_item(item_id)
                self.item_pipelines[item_id] = item_pipeline

    def fit(self):
        """
        Функция для обучения моделей
        """

        for item_id in self.item_ids:
            # # Определяем оптимизатор параметров
            # pipeline_tuner = Tune(
            #     pipeline=self.item_pipelines[item_id],
            #     target_metric=SMAPE(),
            #     horizon=self.horizon
            # )

            # # Находим оптимальное значение параметров
            # best_params = pipeline_tuner.fit(
            #     self.item_data_pools[item_id]["train"],
            #     n_trials=1,
            #     show_progress_bar=False
            # )

            # # Переобучаем модель с новыми параметрами
            # self.item_pipelines[item_id] = self.item_pipelines[item_id].set_params(**best_params)

            # На тюнинг параметров не хватило ресурсов
            self.item_pipelines[item_id].fit(self.item_data_pools[item_id]["train"])

    def predict(self):
        """
        Функция для получения предсказаний
        """

        # Если предсказания уже делали, то просто возвращаем их
        if self.predictions:
            return self.predictions

        # Делаем предсказания и сохраняем их
        predictions = {
            item_id: self.item_pipelines[item_id].forecast()
            for item_id in self.item_ids
        }
        self.predictions = predictions

        return predictions

    def eval(self):
        """
        Функция для оценки качества моделей
        """

        items_metrics = []

        predictions = self.predict()

        for item_id, item_predictions in predictions.items():
            item_metrics_dict = {"item_id": item_id}

            for metric in self.metrics:
                item_metrics_dict[metric.__name__] = metric(mode="macro")(
                    y_true=self.item_data_pools[item_id]["test"],
                    y_pred=item_predictions,
                )
            items_metrics.append(item_metrics_dict)

        metrics_df = pd.DataFrame(items_metrics)

        return metrics_df

    def plot_predictions(self):
        """
        Функция для визуализации предсказаний
        """
        for item_id in self.item_ids:
            if self.use_fitted:
                plot_forecast(forecast_ts=self.predictions[item_id])
            else:
                plot_forecast(
                    forecast_ts=self.predictions[item_id],
                    train_ts=self.item_data_pools[item_id]["train"],
                    test_ts=self.item_data_pools[item_id]["test"],
                    n_train_samples=100,
                )

    def _create_path(self, path):
        Path(path).mkdir(exist_ok=True, parents=True)
        return path

    def _save_artifacts(self):
        self._create_path("models")

        with open("models/item_ids.json", "w") as fp:
            json.dump({"item_ids": self.item_ids}, fp=fp)

        for item_id in self.item_ids:
            self._create_path(f"models/{item_id}")

            self.item_data_pools[item_id]["train"].to_pandas(
                flatten=True
            ).reset_index().to_csv(f"models/{item_id}/train_data.csv", index=False)
            self.item_data_pools[item_id]["test"].to_pandas(
                flatten=True
            ).reset_index().to_csv(f"models/{item_id}/test_data.csv", index=False)

    def _load_artifacts(self):
        with open("models/item_ids.json", "r") as fp:
            self.item_ids = json.load(fp)["item_ids"]

        for item_id in self.item_ids:
            train_data = pd.read_csv(f"models/{item_id}/train_data.csv")
            test_data = pd.read_csv(f"models/{item_id}/test_data.csv")

            train_data["timestamp"] = pd.to_datetime(train_data["timestamp"])
            test_data["timestamp"] = pd.to_datetime(test_data["timestamp"])

            self.item_data_pools[item_id] = {
                "train": TSDataset(df=TSDataset.to_dataset(train_data), freq="D"),
                "test": TSDataset(df=TSDataset.to_dataset(test_data), freq="D"),
            }

    def save_models(self):
        self._save_artifacts()

        for item_id in self.item_ids:
            Path(MODELS_DIR, item_id).mkdir(parents=True, exist_ok=True)
            self.item_pipelines[item_id].save(
                Path(MODELS_DIR, item_id, f"{self.model_name}@{self.horizon}.zip")
            )

    def load_models(self):
        self._load_artifacts()

        for item_id in self.item_ids:
            self.item_pipelines[item_id] = load(
                Path(MODELS_DIR, item_id, f"{self.model_name}@{self.horizon}.zip"),
                ts=self.item_data_pools[item_id]["test"],
            )
