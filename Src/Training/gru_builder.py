# src/training/gru_builder.py

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    GRU,
    Dense,
    Dropout,
    RepeatVector,
    TimeDistributed
)


class GRUBuilder:

    def build(
            self,
            input_steps,
            n_features,
            output_steps
    ):

        model = Sequential()

        model.add(

            GRU(

                128,

                input_shape=(

                    input_steps,

                    n_features

                )

            )

        )

        model.add(
            Dropout(
                0.2
            )
        )

        model.add(
            RepeatVector(
                output_steps
            )
        )

        model.add(

            GRU(

                128,

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