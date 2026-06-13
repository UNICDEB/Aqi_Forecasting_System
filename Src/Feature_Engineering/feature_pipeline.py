# src/feature_engineering/feature_pipeline.py

from Src.Feature_Engineering.scaler import AQIScaler
from Src.Feature_Engineering.sequence_builder import SequenceBuilder
from Src.Feature_Engineering.dataset_builder import DatasetBuilder


class FeaturePipeline:

    def run(self, df):

        features = [

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

        feature_df = df[features]

        scaler = AQIScaler()

        scaled_data = scaler.fit_transform(
            feature_df
        )

        scaler.save(
            "models/scalers/global_scaler.pkl"
        )

        sequence_builder = SequenceBuilder()

        X, y = sequence_builder.create_sequences(
            scaled_data,
            input_window=60,
            output_window=60
        )

        dataset_builder = DatasetBuilder()

        return dataset_builder.split(
            X,
            y
        )