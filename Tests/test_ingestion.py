# # test_ingestion.py

# from Src.Ingestion.ingestion_pipeline import IngestionPipeline

# pipeline = IngestionPipeline(
#     raw_data_path="Dataset/Raw_Data",
#     clean_data_path="Dataset/Clean_Data"
# )

# pipeline.run()

# print("Ingestion Completed")

################################################

import os
import sys

# Finds the root directory (one level up from the 'Tests' folder)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Now Python can see the 'Src' directory
from Src.Ingestion.ingestion_pipeline import IngestionPipeline

pipeline = IngestionPipeline(
    raw_data_path="Dataset/Raw_Data", 
    clean_data_path="Dataset/Clean_Data"
)
pipeline.run()
print("Ingestion Completed")
