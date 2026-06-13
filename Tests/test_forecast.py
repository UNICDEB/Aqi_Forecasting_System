import os
import sys

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from Src.Model_Management.load_model import ModelLoader
from Src.Feature_Engineering.scaler import AQIScaler
from Src.Forecasting.forecast_pipeline import ForecastPipeline


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


def main():
    data_path = os.path.join(
        project_root,
        "Dataset",
        "Clean_Data",
        "MCT2408020_processed.csv"
    )

    model_path = os.path.join(
        project_root,
        "Models",
        "LSTM_TEST",
        "best_model.keras"
    )

    scaler_path = os.path.join(
        project_root,
        "Models",
        "Scalers",
        "global_scaler.pkl"
    )

    df = pd.read_csv(data_path)

    loader = ModelLoader()
    model = loader.load(model_path)

    scaler = AQIScaler()
    scaler.load(scaler_path)

    latest_data = scaler.transform(df[FEATURES])
    latest_sequence = latest_data[-60:]

    pipeline = ForecastPipeline()
    valid_timestamps = pd.to_datetime(
        df["TIMESTAMP"],
        errors="coerce"
    ).dropna()

    forecast_df = pipeline.run(
        model,
        scaler.scaler,
        latest_sequence,
        valid_timestamps.iloc[-1],
        "MCT2408020"
    )

    print(forecast_df.head())


if __name__ == "__main__":
    main()