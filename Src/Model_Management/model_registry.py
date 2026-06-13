# src/model_management/model_registry.py

from pathlib import Path


class ModelRegistry:

    def get_models(self):

        models_dir = Path(
            "models"
        )

        available = []

        for item in models_dir.iterdir():

            if item.is_dir():

                available.append(
                    item.name
                )

        return available