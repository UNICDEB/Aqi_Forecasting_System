# # test_gru_training.py

# import pandas as pd

# from Src.Feature_Engineering.feature_pipeline import (
#     FeaturePipeline
# )

# from Src.Training.train_gru import (
#     GRUTrainingPipeline
# )

# from Src.Evaluation.evaluation_pipeline import (
#     EvaluationPipeline
# )

# from Src.Model_Management.save_model import (
#     ModelSaver
# )

# print("=" * 60)
# print("Loading Dataset")
# print("=" * 60)

# df = pd.read_csv(
#     "dataset/Clean_Data/MCT2408020_processed.csv"
# )

# pipeline = FeaturePipeline()

# (
#     X_train,
#     X_val,
#     X_test,
#     y_train,
#     y_val,
#     y_test,
#     scaler
# ) = pipeline.run(df)

# print("\nDataset Shapes")

# print("X_train:", X_train.shape)
# print("y_train:", y_train.shape)

# print("X_val:", X_val.shape)
# print("y_val:", y_val.shape)

# print("X_test:", X_test.shape)
# print("y_test:", y_test.shape)

# # Optional for quick testing
# # X_train = X_train[:5000]
# # y_train = y_train[:5000]
# # X_val = X_val[:1000]
# # y_val = y_val[:1000]

# print("\nStarting GRU Training...")

# trainer = GRUTrainingPipeline()

# model, history = trainer.run(
#     X_train,
#     y_train,
#     X_val,
#     y_val
# )

# print("\nTraining Completed")

# print("\nEvaluating Model...")

# evaluation = EvaluationPipeline()

# predictions, metrics_df = evaluation.run(
#     model,
#     X_test,
#     y_test,
#     scaler,
#     model_name="GRU"
# )

# print(metrics_df)

# print("\nSaving Model...")

# saver = ModelSaver()

# saver.save(
#     model,
#     history,
#     metrics_df,
#     model_name="GRU"
# )

# print("\nGRU Pipeline Finished Successfully")


######################################

import os
import sys
import pandas as pd

# 1. PATH FIX: This must be at the very top before any imports from Src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 2. Now you can safely import from your Src package
from Src.Feature_Engineering.feature_pipeline import FeaturePipeline
from Src.Feature_Engineering.scaler import AQIScaler
from Src.Training.train_gru import GRUTrainingPipeline
from Src.Evaluation.evaluation_pipeline import EvaluationPipeline
from Src.Model_Management.save_model import ModelSaver

def main():
    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    df = pd.read_csv(
        "Dataset/Clean_Data/MCT2408020_processed.csv",
        low_memory=False
    )

    pipeline = FeaturePipeline()
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = pipeline.run(df)

    scaler = AQIScaler()
    scaler.load("Models/Scalers/global_scaler.pkl")

    print("\nDataset Shapes")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # Optional for quick testing
    # X_train = X_train[:5000]
    # y_train = y_train[:5000]
    # X_val = X_val[:1000]
    # y_val = y_val[:1000]

    print("\nStarting GRU Training...")
    trainer = GRUTrainingPipeline()
    model, history = trainer.run(
        X_train,
        y_train,
        X_val,
        y_val
    )
    print("\nTraining Completed")

    print("\nEvaluating Model...")
    evaluation = EvaluationPipeline()
    predictions, metrics_df = evaluation.run(
        model,
        X_test,
        y_test,
        scaler,
        model_name="GRU"
    )
    print(metrics_df)

    print("\nSaving Model...")
    saver = ModelSaver()
    saver.save(
        model,
        history,
        metrics_df,
        model_name="GRU"
    )

    print("\nGRU Pipeline Finished Successfully")

if __name__ == "__main__":
    main()
