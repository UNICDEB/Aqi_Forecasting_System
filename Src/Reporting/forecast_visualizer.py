# src/reporting/forecast_visualizer.py

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


class ForecastVisualizer:

    def plot_aqi(
            self,
            forecast_file
    ):

        df = pd.read_excel(
            forecast_file
        )

        output_dir = Path(
            "Reports/forecasting"
        )

        output_dir.mkdir(
            exist_ok=True,
            parents=True
        )

        plt.figure(
            figsize=(12,6)
        )

        plt.plot(
            df["AQI"]
        )

        plt.title(
            "Forecast AQI"
        )

        plt.xlabel(
            "Minutes"
        )

        plt.ylabel(
            "AQI"
        )

        plt.savefig(
            output_dir /
            "AQI_forecast.png"
        )

        plt.close()