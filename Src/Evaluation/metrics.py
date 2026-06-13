# from sklearn.metrics import (
#     mean_squared_error,
#     mean_absolute_error,
#     r2_score
# )

# import numpy as np


# class Metrics:

#     @staticmethod
#     def calculate(
#             y_true,
#             y_pred
#     ):

#         y_true = y_true.reshape(-1)

#         y_pred = y_pred.reshape(-1)

#         rmse = np.sqrt(
#             mean_squared_error(
#                 y_true,
#                 y_pred
#             )
#         )

#         mae = mean_absolute_error(
#             y_true,
#             y_pred
#         )

#         r2 = r2_score(
#             y_true,
#             y_pred
#         )

#         mape = np.mean(
#             np.abs(
#                 (
#                     y_true -
#                     y_pred
#                 ) /
#                 (
#                     y_true + 1e-8
#                 )
#             )
#         ) * 100

#         return {
#             "RMSE": rmse,
#             "MAE": mae,
#             "MAPE": mape,
#             "R2": r2
#         }

###################

# src/evaluation/metrics.py
## Updated Version of the Metrics class with improved structure and error handling.

import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


class Metrics:

    @staticmethod
    def calculate(
            y_true,
            y_pred
    ):

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        r2 = r2_score(
            y_true,
            y_pred
        )

        mape = np.mean(
            np.abs(
                (y_true - y_pred)
                /
                (y_true + 1e-8)
            )
        ) * 100

        return {

            "RMSE": float(rmse),

            "MAE": float(mae),

            "MAPE": float(mape),

            "R2": float(r2)

        }