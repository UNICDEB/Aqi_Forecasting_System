# src/ingestion/excel_reader.py

import pandas as pd


class ExcelReader:

    def __init__(self):
        pass

    def read_device_file(self, filepath):

        df = pd.read_excel(
            filepath,
            skiprows=7
        )

        df.columns = [str(col).strip() for col in df.columns]

        return df