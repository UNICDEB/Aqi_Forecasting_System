# src/forecasting/excel_exporter.py

from pathlib import Path
import pandas as pd


class ExcelExporter:

    def save(
            self,
            forecast_df,
            device_id
    ):

        output_dir = Path(
            "forecasts/LSTM"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (

            output_dir /

            f"{device_id}_forecast.xlsx"

        )

        forecast_df.to_excel(
            output_file,
            index=False
        )

        print(
            f"Forecast Saved: {output_file}"
        )