from Src.Ingestion.ingestion_pipeline import (
    IngestionPipeline
)

pipeline = IngestionPipeline(
    raw_data_path="Dataset/Raw_Data",
    clean_data_path="Dataset/Clean_Data"
)

pipeline.run()