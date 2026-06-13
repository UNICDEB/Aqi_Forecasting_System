import numpy as np


class InverseScaler:

    def transform(
            self,
            scaler,
            data
    ):

        samples = data.shape[0]
        steps = data.shape[1]
        features = data.shape[2]

        reshaped = data.reshape(
            -1,
            features
        )

        restored = scaler.inverse_transform(
            reshaped
        )

        restored = restored.reshape(
            samples,
            steps,
            features
        )

        return restored