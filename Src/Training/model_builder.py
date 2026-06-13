# src/training/model_builder.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    RepeatVector,
    TimeDistributed
)


class LSTMModelBuilder:

    def build(
            self,
            input_steps=60,
            n_features=11,
            output_steps=60
    ):

        model = Sequential()

        model.add(
            LSTM(
                128,
                activation="tanh",
                input_shape=(
                    input_steps,
                    n_features
                )
            )
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            RepeatVector(
                output_steps
            )
        )

        model.add(
            LSTM(
                128,
                activation="tanh",
                return_sequences=True
            )
        )

        model.add(
            TimeDistributed(
                Dense(
                    n_features
                )
            )
        )

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )

        return model