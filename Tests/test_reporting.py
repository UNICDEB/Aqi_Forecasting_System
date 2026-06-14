# # test_reporting.py

# from Src.Reporting.training_visualizer import (
#     TrainingVisualizer
# )

# visualizer = TrainingVisualizer()

# visualizer.plot(
#     "Models/LSTM_TEST/training_history.csv"
# )

###############################

import os
import sys

# 1. PATH FIX: This must be at the very top before importing from Src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 2. Safely import from your Src package
from Src.Reporting.training_visualizer import TrainingVisualizer

def main():
    # 3. Initialize visualizer tool
    print("Initializing Training Visualizer...")
    visualizer = TrainingVisualizer()

    # 4. Plot and save metrics from the history CSV file
    history_path = "Models/LSTM_TEST/training_history.csv"
    print(f"Reading training history from: {history_path}")
    
    visualizer.plot(history_path)
    print("\nVisualization generated successfully!")

if __name__ == "__main__":
    main()
