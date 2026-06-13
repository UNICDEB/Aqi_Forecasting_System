# src/forecasting/forecast_pipeline.py

import pandas as pd

from Src.Forecasting.forecast_generator import (
    ForecastGenerator
)

from Src.Forecasting.timestamp_generator import (
    TimestampGenerator
)

from Src.Forecasting.excel_exporter import (
    ExcelExporter
)

from Src.Evaluation.inverse_scaler import (
    InverseScaler
)


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


class ForecastPipeline:

    def run(

            self,

            model,

            scaler,

            latest_sequence,

            last_timestamp,

            device_id="MCT2408020"

    ):

        generator = ForecastGenerator()

        prediction_scaled = generator.predict(
            model,
            latest_sequence
        )

        inverse = InverseScaler()

        prediction = inverse.transform(
            scaler,
            prediction_scaled.reshape(
                1,
                prediction_scaled.shape[0],
                prediction_scaled.shape[1]
            )
        )[0]

        timestamp_gen = TimestampGenerator()

        timestamps = timestamp_gen.generate(
            last_timestamp,
            prediction.shape[0]
        )

        forecast_df = pd.DataFrame(
            prediction,
            columns=FEATURES
        )

        forecast_df.insert(
            0,
            "TIMESTAMP",
            timestamps
        )

        exporter = ExcelExporter()

        exporter.save(
            forecast_df,
            device_id
        )

        return forecast_df