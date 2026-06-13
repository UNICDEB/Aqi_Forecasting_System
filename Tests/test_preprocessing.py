# # test_preprocessing.py

# import pandas as pd

# from Src.Preprocessing.preprocessing_pipeline import (
#     PreprocessingPipeline
# )

# # load one merged file

# df = pd.read_csv(
#     "dataset/Clean_Data/MCT2408020.csv"
# )

# pipeline = PreprocessingPipeline()

# processed_df, report = pipeline.run(df)

# print("\nValidation Report")
# print(report)

# print("\nColumns")
# print(processed_df.columns)

# print("\nShape")
# print(processed_df.shape)

# print("\nFirst Rows")
# print(processed_df.head())

# processed_df.to_csv(
#     "dataset/Clean_Data/MCT2408020_processed.csv",
#     index=False
# )

# print("\nSaved Successfully")

######################

import os
import sys
import pandas as pd

# 1. Path Fix: Allows Python to see the 'Src' folder from the 'Tests' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 2. Imports from Src
from Src.Preprocessing.preprocessing_pipeline import PreprocessingPipeline

def main():
    # 3. Load clean data
    input_path = "Dataset/Clean_Data/MCT2408020.csv"
    print(r"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    # 4. Initialize and run pipeline
    pipeline = PreprocessingPipeline()
    processed_df, report = pipeline.run(df)

    # 5. Print pipeline results
    print("\n=== Validation Report ===")
    print(report)

    print("\n=== Processed Data Details ===")
    print(f"Columns: {list(processed_df.columns)}")
    print(f"Shape: {processed_df.shape}")
    
    print("\n=== First 5 Rows ===")
    print(processed_df.head())

    # 6. Save processed data
    output_path = "Dataset/Clean_Data/MCT2408020_processed.csv"
    processed_df.to_csv(output_path, index=False)
    print(f"\nSaved Successfully to: {output_path}")

if __name__ == "__main__":
    main()
