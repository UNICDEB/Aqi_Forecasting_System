# src/reporting/training_visualizer.py

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


class TrainingVisualizer:

    def plot(self, history_file):

        history = pd.read_csv(
            history_file
        )

        output_dir = Path(
            "Reports/training"
        )

        output_dir.mkdir(
            exist_ok=True,
            parents=True
        )

        # Loss Curve
        plt.figure(figsize=(10,5))

        plt.plot(
            history["loss"],
            label="Train"
        )

        plt.plot(
            history["val_loss"],
            label="Validation"
        )

        plt.title(
            "Training Loss"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Loss"
        )

        plt.legend()

        plt.savefig(
            output_dir /
            "loss_curve.png"
        )

        plt.close()

        print(
            "Loss curve saved."
        )