import os
import sys

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from Src.Model_Management.load_model import ModelLoader
from Src.Feature_Engineering.scaler import AQIScaler
from Src.Forecasting.recursive_forecaster import (
    RecursiveForecaster
)

from Src.Forecasting.timestamp_generator import (
    TimestampGenerator
)

from Src.Forecasting.summary_generator import (
    SummaryGenerator
)

from Src.Forecasting.forecast_validator import (
    ForecastValidator
)

from Src.Evaluation.inverse_scaler import (
    InverseScaler
)

# ==========================================
# USER INPUT
# ==========================================

FORECAST_DAYS = 7

FORECAST_HOURS = 12

DEVICE_ID = "MCT2408020"

# ==========================================
# FEATURES
# ==========================================

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

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    os.path.join(
        project_root,
        "Dataset",
        "Clean_Data",
        "MCT2408020_processed.csv"
    )
)

df["TIMESTAMP"] = pd.to_datetime(
    df["TIMESTAMP"],
    errors="coerce"
)

# ==========================================
# LOAD MODEL
# ==========================================

loader = ModelLoader()

model = loader.load(
    os.path.join(
        project_root,
        "Models",
            "LSTM_TEST",
        "best_model.keras"
    )
)

# ==========================================
# LOAD SCALER
# ==========================================

scaler = AQIScaler()

scaler.load(
    os.path.join(
        project_root,
        "Models",
        "Scalers",
        "global_scaler.pkl"
    )
)

# ==========================================
# PREPARE INPUT
# ==========================================

scaled_data = scaler.transform(
    df[FEATURES]
)

latest_sequence = scaled_data[-60:]

last_timestamp = df[
    "TIMESTAMP"
].dropna().iloc[-1]

# ==========================================
# FORECAST DAYS
# ==========================================

day_minutes = (
    FORECAST_DAYS *
    24 *
    60
)

recursive = RecursiveForecaster()

day_forecast_scaled = recursive.forecast(
    model,
    latest_sequence,
    day_minutes
)

inverse = InverseScaler()

day_forecast = inverse.transform(
    scaler.scaler,
    day_forecast_scaled.reshape(
        1,
        day_forecast_scaled.shape[0],
        day_forecast_scaled.shape[1]
    )
)[0]

# ==========================================
# TIMESTAMP GENERATION
# ==========================================

timestamp_gen = TimestampGenerator()

day_timestamps = timestamp_gen.generate(
    last_timestamp,
    len(day_forecast)
)

day_forecast_df = pd.DataFrame(
    day_forecast,
    columns=FEATURES
)

day_forecast_df.insert(
    0,
    "TIMESTAMP",
    day_timestamps
)

# ==========================================
# VALIDATE
# ==========================================

validator = ForecastValidator()

day_forecast_df = validator.validate(
    day_forecast_df
)

# ==========================================
# SAVE DAY FORECAST
# ==========================================

day_output = (
    f"Forecasts/LSTM/"
    f"{DEVICE_ID}_{FORECAST_DAYS}days.xlsx"
)

day_forecast_df.to_excel(
    day_output,
    index=False
)

print()
print("=" * 60)
print("DAY FORECAST SAVED")
print(day_output)
print("=" * 60)

# ==========================================
# FORECAST HOURS
# ==========================================

hour_minutes = (
    FORECAST_HOURS *
    60
)

hour_forecast_scaled = recursive.forecast(
    model,
    latest_sequence,
    hour_minutes
)

hour_forecast = inverse.transform(
    scaler.scaler,
    hour_forecast_scaled.reshape(
        1,
        hour_forecast_scaled.shape[0],
        hour_forecast_scaled.shape[1]
    )
)[0]

hour_timestamps = timestamp_gen.generate(
    last_timestamp,
    len(hour_forecast)
)

hour_forecast_df = pd.DataFrame(
    hour_forecast,
    columns=FEATURES
)

hour_forecast_df.insert(
    0,
    "TIMESTAMP",
    hour_timestamps
)

hour_forecast_df = validator.validate(
    hour_forecast_df
)

hour_output = (
    f"Forecasts/LSTM/"
    f"{DEVICE_ID}_{FORECAST_HOURS}hours.xlsx"
)

hour_forecast_df.to_excel(
    hour_output,
    index=False
)

print()
print("=" * 60)
print("HOUR FORECAST SAVED")
print(hour_output)
print("=" * 60)

# ==========================================
# SUMMARY
# ==========================================

summary = SummaryGenerator()

daily_summary = summary.daily_summary(
    day_forecast_df
)

hourly_summary = summary.hourly_summary(
    hour_forecast_df
)

daily_summary.to_excel(
    f"Forecasts/LSTM/"
    f"{DEVICE_ID}_daily_summary.xlsx"
)

hourly_summary.to_excel(
    f"Forecasts/LSTM/"
    f"{DEVICE_ID}_hourly_summary.xlsx"
)

# ==========================================
# PRINT SUMMARY
# ==========================================

print()
print("=" * 60)
print("DAILY SUMMARY")
print("=" * 60)

print(
    daily_summary.head()
)

print()

print("=" * 60)
print("HOURLY SUMMARY")
print("=" * 60)

print(
    hourly_summary.head()
)

print()

print("=" * 60)
print("FORECAST COMPLETE")
print("=" * 60)