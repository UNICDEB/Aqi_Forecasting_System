# src/ingestion/folder_scanner.py

from pathlib import Path


class FolderScanner:

    def __init__(self, raw_data_path):
        self.raw_data_path = Path(raw_data_path)

    def get_date_folders(self):
        folders = [
            folder for folder in self.raw_data_path.iterdir()
            if folder.is_dir()
        ]

        folders.sort()

        return folders

    def get_all_excel_files(self):

        excel_files = []

        for folder in self.get_date_folders():

            files = list(folder.glob("*.xlsx"))

            excel_files.extend(files)

        return excel_files