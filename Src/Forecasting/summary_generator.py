# summary_generator.py

import pandas as pd


class SummaryGenerator:

    def hourly_summary(
            self,
            forecast_df
    ):

        temp = forecast_df.copy()

        temp["HOUR"] = (
            temp["TIMESTAMP"]
            .dt.floor("H")
        )

        return temp.groupby(
            "HOUR"
        ).agg(
            ["mean", "min", "max"]
        )

    def daily_summary(
            self,
            forecast_df
    ):

        temp = forecast_df.copy()

        temp["DAY"] = (
            temp["TIMESTAMP"]
            .dt.date
        )

        return temp.groupby(
            "DAY"
        ).agg(
            ["mean", "min", "max"]
        )