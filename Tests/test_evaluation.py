# # test_evaluation.py

# from Src.Evaluation.evaluator import (
#     Evaluator
# )

# evaluator = Evaluator()

# predictions, metrics = evaluator.evaluate(
#     model,
#     X_test,
#     y_test
# )

# print(metrics)


###################

# import os
# import sys

# import pandas as pd
# from tensorflow.keras.models import load_model

# # 1. PATH FIX: This must be at the very top before importing from Src
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(project_root)

# # 2. Now you can safely import from Src
# from Src.Evaluation.evaluator import Evaluator
# from Src.Evaluation.metrics import Metrics
# from Src.Feature_Engineering.dataset_builder import DatasetBuilder
# from Src.Feature_Engineering.scaler import AQIScaler
# from Src.Feature_Engineering.sequence_builder import SequenceBuilder

# def main():
#     print("Initializing evaluator...")

#     model_path = os.path.join(
#         project_root,
#         "Models",
#         "LSTM",
#         "best_model.keras"
#     )

#     data_path = os.path.join(
#         project_root,
#         "Dataset",
#         "Clean_Data",
#         "MCT2408020_processed.csv"
#     )

#     scaler_path = os.path.join(
#         project_root,
#         "Models",
#         "Scalers",
#         "global_scaler.pkl"
#     )

#     model = load_model(model_path)

#     df = pd.read_csv(data_path)

#     features = [
#         "PM10",
#         "PM25",
#         "PM1",
#         "NO2",
#         "SO2",
#         "CO",
#         "O3",
#         "CO2",
#         "TEMP",
#         "HUMIDITY",
#         "AQI"
#     ]

#     feature_df = df[features]

#     scaler = AQIScaler()
#     scaler.load(scaler_path)
#     scaled_data = scaler.transform(feature_df)

#     sequence_builder = SequenceBuilder()
#     X, y = sequence_builder.create_sequences(
#         scaled_data,
#         input_window=60,
#         output_window=60
#     )

#     dataset_builder = DatasetBuilder()
#     _, _, X_test, _, _, y_test = dataset_builder.split(X, y)

#     evaluator = Evaluator()

#     predictions, metrics = evaluator.evaluate(
#         model,
#         X_test,
#         y_test
#     )

#     y_test_original = scaler.inverse_transform(
#         y_test.reshape(
#             -1,
#             11
#         )
#     )

#     pred_original = scaler.inverse_transform(
#         predictions.reshape(
#             -1,
#             11
#         )
#     )

#     original_metrics = Metrics.calculate(
#         y_test_original,
#         pred_original
#     )

#     print("\n=== Evaluation Metrics (Scaled) ===")
#     print(metrics)

#     print("\n=== Evaluation Metrics (Original Scale) ===")
#     print(original_metrics)

# if __name__ == "__main__":
#     main()


##########################################

import os
import sys

import pandas as pd
from tensorflow.keras.models import load_model

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from Src.Evaluation.evaluation_pipeline import EvaluationPipeline
from Src.Feature_Engineering.dataset_builder import DatasetBuilder
from Src.Feature_Engineering.scaler import AQIScaler
from Src.Feature_Engineering.sequence_builder import SequenceBuilder


def main():
    print("Initializing evaluator...")

    model_path = os.path.join(
        project_root,
        "Models",
        "LSTM",
        "best_model.keras"
    )

    data_path = os.path.join(
        project_root,
        "Dataset",
        "Clean_Data",
        "MCT2408020_processed.csv"
    )

    scaler_path = os.path.join(
        project_root,
        "Models",
        "Scalers",
        "global_scaler.pkl"
    )

    model = load_model(model_path)

    df = pd.read_csv(data_path)

    features = [
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

    feature_df = df[features]

    scaler = AQIScaler()
    scaler.load(scaler_path)
    scaled_data = scaler.transform(feature_df)

    sequence_builder = SequenceBuilder()
    X, y = sequence_builder.create_sequences(
        scaled_data,
        input_window=60,
        output_window=60
    )

    dataset_builder = DatasetBuilder()
    _, _, X_test, _, _, y_test = dataset_builder.split(X, y)

    pipeline = EvaluationPipeline()
    predictions, metrics_df = pipeline.run(
        model,
        X_test,
        y_test,
        model_name="LSTM"
    )

    print("\n=== Evaluation Metrics ===")
    print(metrics_df)

    print("\nSaved Excel report to: Reports/Metrics/LSTM_metrics.xlsx")


if __name__ == "__main__":
    main()