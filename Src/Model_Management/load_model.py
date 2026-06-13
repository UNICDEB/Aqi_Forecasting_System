# src/model_management/load_model.py

from tensorflow.keras.models import (
    load_model
)


class ModelLoader:

    def load(
            self,
            model_path
    ):

        model = load_model(
            model_path
        )

        return model