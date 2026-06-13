

# src/evaluation/parameter_metrics.py
## Code for calculating metrics for each parameter in the dataset.

import pandas as pd

from Src.Evaluation.metrics import Metrics


FEATURES = [

    "PM10",
    "PM25",
    "PM1",

    "NO2",
    "SO2",

    "CO",

    "O3",

    "CO2",

    "TEMP",
    "HUMIDITY",

    "AQI"
]


class ParameterMetrics:

    def calculate(

            self,

            y_true,

            y_pred

    ):

        results = []

        for idx, feature in enumerate(FEATURES):

            true_values = y_true[:, :, idx].flatten()

            pred_values = y_pred[:, :, idx].flatten()

            metrics = Metrics.calculate(

                true_values,

                pred_values

            )

            metrics["Parameter"] = feature

            results.append(metrics)

        return pd.DataFrame(results)