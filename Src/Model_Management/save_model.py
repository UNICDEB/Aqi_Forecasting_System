# src/model_management/save_model.py

from pathlib import Path
import pandas as pd


class ModelSaver:

    def save(
            self,
            model,
            history,
            metrics_df,
            model_name="LSTM"
    ):

        model_dir = Path(
            f"Models/{model_name}"
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save Model

        model.save(
            model_dir /
            "best_model.keras"
        )

        # Save History

        history_df = pd.DataFrame(
            history.history
        )

        history_df.to_csv(
            model_dir /
            "training_history.csv",
            index=False
        )

        # Save Metrics

        metrics_df.to_excel(
            model_dir /
            "metrics.xlsx",
            index=False
        )

        print(
            f"{model_name} saved successfully"
        )