import pandas as pd

class TimestampProcessor:

    def process(self, df):

        df["TIMESTAMP"] = pd.to_datetime(
            df["TIMESTAMP"],
            errors="coerce"
        )

        df = df.sort_values(
            "TIMESTAMP"
        )

        df["YEAR"] = df["TIMESTAMP"].dt.year

        df["MONTH"] = df["TIMESTAMP"].dt.month

        df["DAY"] = df["TIMESTAMP"].dt.day

        df["HOUR"] = df["TIMESTAMP"].dt.hour

        df["MINUTE"] = df["TIMESTAMP"].dt.minute

        df["WEEKDAY"] = df["TIMESTAMP"].dt.weekday

        return df