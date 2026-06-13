# src/ingestion/ingestion_pipeline.py

from collections import defaultdict
from pathlib import Path

from Src.Ingestion.folder_scanner import FolderScanner
from Src.Ingestion.device_merger import DeviceMerger


class IngestionPipeline:

    def __init__(
            self,
            raw_data_path,
            clean_data_path
    ):

        self.raw_data_path = raw_data_path
        self.clean_data_path = Path(clean_data_path)

        self.clean_data_path.mkdir(
            exist_ok=True,
            parents=True
        )

    def run(self):

        scanner = FolderScanner(
            self.raw_data_path
        )

        excel_files = scanner.get_all_excel_files()

        device_map = defaultdict(list)

        for file in excel_files:

            device_id = file.stem

            device_map[device_id].append(file)

        merger = DeviceMerger()

        for device_id, files in device_map.items():

            merged_df = merger.merge_device_data(
                files
            )

            output_file = (
                self.clean_data_path /
                f"{device_id}.csv"
            )

            merged_df.to_csv(
                output_file,
                index=False
            )

            print(
                f"Saved : {device_id}"
            )