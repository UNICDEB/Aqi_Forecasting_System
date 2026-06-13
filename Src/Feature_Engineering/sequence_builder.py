# src/feature_engineering/sequence_builder.py

import numpy as np


class SequenceBuilder:

    def create_sequences(
            self,
            data,
            input_window=60,
            output_window=60
    ):

        X = []
        y = []

        total_length = len(data)

        for i in range(
                total_length -
                input_window -
                output_window
        ):

            X.append(

                data[
                i:
                i + input_window
                ]

            )

            y.append(

                data[
                i + input_window:
                i + input_window + output_window
                ]

            )

        return np.array(X), np.array(y)