import os
import sys
import pandas as pd

# 1. Path Fix: Allows Python to see the 'src' folder from the 'Tests' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 2. Imports from src
from Src.Feature_Engineering.feature_pipeline import FeaturePipeline

def main():
    # 3. Load processed data
    input_path = "Dataset/Clean_Data/MCT2408020_processed.csv"
    print(f"Loading processed data from: {input_path}")
    df = pd.read_csv(input_path)

    # 4. Initialize and run pipeline
    pipeline = FeaturePipeline()
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.run(df)

    # 5. Print Split Statistics Cleanly
    print("\n" + "="*30)
    print(" FEATURE ENGINEERING SPLITS ")
    print("="*30)
    
    print(f"X_train shape : {X_train.shape}")
    print(f"y_train shape : {y_train.shape}")
    print("-" * 30)
    
    print(f"X_val shape   : {X_val.shape}")
    print(f"y_val shape   : {y_val.shape}")
    print("-" * 30)
    
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_test shape  : {y_test.shape}")
    print("="*30)

if __name__ == "__main__":
    main()
