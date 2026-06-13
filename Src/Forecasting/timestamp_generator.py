# src/forecasting/timestamp_generator.py

import pandas as pd


class TimestampGenerator:

    def generate(
            self,
            last_timestamp,
            steps
    ):

        timestamps = pd.date_range(
            start=last_timestamp +
                  pd.Timedelta(minutes=1),
            periods=steps,
            freq="min"
        )

        return timestamps