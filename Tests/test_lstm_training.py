# # test_lstm_training.py

# import pandas as pd

# from Src.Feature_Engineering.feature_pipeline import (
#     FeaturePipeline
# )

# from Src.Training.train_lstm import (
#     LSTMTrainingPipeline
# )

# df = pd.read_csv(
#     "dataset/Clean_Data/MCT2408020_processed.csv"
# )

# feature_pipeline = FeaturePipeline()

# (
#     X_train,
#     X_val,
#     X_test,

#     y_train,
#     y_val,
#     y_test

# ) = feature_pipeline.run(df)

# trainer = LSTMTrainingPipeline()

# model, history = trainer.run(

#     X_train,
#     y_train,

#     X_val,
#     y_val
# )

# print()

# print("Training Completed")

#################################

import os
import sys
import pandas as pd

# 1. PATH FIX: This MUST be at the very top before importing from Src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 2. Now you can safely import from Src
from Src.Feature_Engineering.feature_pipeline import FeaturePipeline
from Src.Training.train_lstm import LSTMTrainingPipeline

def main():
    # Load dataset
    df = pd.read_csv("dataset/Clean_Data/MCT2408020_processed.csv")

    # Run feature engineering
    feature_pipeline = FeaturePipeline()
    X_train, X_val, X_test, y_train, y_val, y_test = feature_pipeline.run(df)

    # Run training
    trainer = LSTMTrainingPipeline()
    model, history = trainer.run(
        X_train,
        y_train,
        X_val,
        y_val
    )

    print("\nTraining Completed")

if __name__ == "__main__":
    main()



###########################################
#### Other Versions of the test file (with different import paths) are available in the edit history. Please refer to the edit history for more details.

# import os
# import sys
# import pandas as pd

# # 1. Path Fix: Allows Python to locate the 'Src' directory from inside 'Tests'
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(project_root)

# # 2. Imports from Src Package
# from Src.Feature_Engineering.feature_pipeline import FeaturePipeline
# from Src.Training.train_lstm import LSTMTrainingPipeline

# def main():
#     # 3. Load preprocessed feature dataset
#     input_path = "Dataset/Clean_Data/MCT2408020_processed.csv"
#     print(f"Loading data from: {input_path}")
#     df = pd.read_csv(input_path)

#     # 4. Extract and split features
#     print("\nRunning feature engineering pipeline...")
#     feature_pipeline = FeaturePipeline()
#     X_train, X_val, X_test, y_train, y_val, y_test = feature_pipeline.run(df)

#     # 5. Initialize and run LSTM Model training loop
#     print("\nInitializing LSTM training pipeline...")
#     trainer = LSTMTrainingPipeline()
#     model, history = trainer.run(
#         X_train,
#         y_train,
#         X_val,
#         y_val
#     )

#     print("\n" + "="*30)
#     print(" Training Completed Successfully! ")
#     print("="*30)

# if __name__ == "__main__":
#     main()
