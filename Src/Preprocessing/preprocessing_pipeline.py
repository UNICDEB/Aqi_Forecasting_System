from Src.Preprocessing.cleaner import Cleaner
from Src.Preprocessing.outlier_detector import OutlierDetector
from Src.Preprocessing.timestamp_processor import TimestampProcessor
from Src.Preprocessing.validator import Validator
from Src.Preprocessing.column_mapper import COLUMN_MAPPING

class PreprocessingPipeline:

    def run(self, df):

        # Rename columns

        df.rename(
            columns= COLUMN_MAPPING,
            inplace=True
        )

        validator = Validator()

        report = validator.validate(df)

        timestamp_processor = TimestampProcessor()

        df = timestamp_processor.process(df)

        cleaner = Cleaner()

        df = cleaner.clean(df)

        outlier = OutlierDetector()

        target_columns = [

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

        df = outlier.remove_outliers(
            df,
            target_columns
        )

        return df, report