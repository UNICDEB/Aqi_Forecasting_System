import os
import sys

import pandas as pd
from tensorflow.keras.models import load_model

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from Src.Model_Management.save_model import ModelSaver


class DummyHistory:
    def __init__(self):
        self.history = {
            "loss": [0.42, 0.31, 0.24],
            "val_loss": [0.48, 0.35, 0.29]
        }


def main():
    model_path = os.path.join(
        project_root,
        "Models",
        "LSTM",
        "best_model.keras"
    )

    metrics_path = os.path.join(
        project_root,
        "Reports",
        "Metrics",
        "LSTM_metrics.xlsx"
    )

    model = load_model(model_path)
    history = DummyHistory()
    metrics_df = pd.read_excel(metrics_path)

    saver = ModelSaver()
    saver.save(
        model,
        history,
        metrics_df,
        model_name="LSTM_TEST"
    )

    print("Model save test completed successfully")


if __name__ == "__main__":
    main()