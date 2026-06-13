import numpy as np

class Cleaner:

    def clean(self, df):

        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        numeric_columns = df.select_dtypes(
            include=["number"]
        ).columns

        for col in numeric_columns:

            df[col] = df[col].interpolate()

            df[col] = df[col].fillna(
                method="bfill"
            )

            df[col] = df[col].fillna(
                method="ffill"
            )

        return df