import numpy as np


class RecursiveForecaster:

    def forecast(
            self,
            model,
            initial_sequence,
            total_minutes
    ):

        current_sequence = initial_sequence.copy()

        forecasts = []

        iterations = total_minutes // 60

        remainder = total_minutes % 60

        for _ in range(iterations):

            prediction = model.predict(
                np.expand_dims(
                    current_sequence,
                    axis=0
                ),
                verbose=0
            )[0]

            forecasts.extend(
                prediction
            )

            current_sequence = prediction

        if remainder > 0:

            prediction = model.predict(
                np.expand_dims(
                    current_sequence,
                    axis=0
                ),
                verbose=0
            )[0]

            forecasts.extend(
                prediction[:remainder]
            )

        return np.array(
            forecasts
        )