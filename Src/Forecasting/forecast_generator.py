# src/forecasting/forecast_generator.py

import numpy as np


class ForecastGenerator:

    def predict(
            self,
            model,
            input_sequence
    ):

        prediction = model.predict(
            np.expand_dims(
                input_sequence,
                axis=0
            ),
            verbose=0
        )

        return prediction[0]