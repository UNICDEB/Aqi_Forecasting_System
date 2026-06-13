class OutlierDetector:

    def remove_outliers(
            self,
            df,
            columns
    ):

        for col in columns:

            q1 = df[col].quantile(0.25)

            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr

            upper = q3 + 1.5 * iqr

            df[col] = df[col].clip(
                lower,
                upper
            )

        return df