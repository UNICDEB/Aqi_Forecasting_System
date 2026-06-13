# src/forecasting/forecast_validator.py

import pandas as pd


class ForecastValidator:

    def validate(self, forecast_df: pd.DataFrame):

        df = forecast_df.copy()

        # Air pollutants cannot be negative
        pollutant_cols = [
            "PM10",
            "PM25",
            "PM1",
            "NO2",
            "SO2",
            "CO",
            "O3",
            "CO2",
            "AQI"
        ]

        for col in pollutant_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        # Temperature range
        if "TEMP" in df.columns:
            df["TEMP"] = df["TEMP"].clip(
                lower=-10,
                upper=60
            )

        # Humidity range
        if "HUMIDITY" in df.columns:
            df["HUMIDITY"] = df["HUMIDITY"].clip(
                lower=0,
                upper=100
            )

        return df