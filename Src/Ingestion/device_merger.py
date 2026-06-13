# src/ingestion/device_merger.py

import pandas as pd
from pathlib import Path


class DeviceMerger:

    def __init__(self):
        pass

    def merge_device_data(
            self,
            device_files
    ):

        all_df = []

        for file in device_files:

            df = pd.read_excel(
                file,
                skiprows=7
            )

            all_df.append(df)

        merged_df = pd.concat(
            all_df,
            ignore_index=True
        )

        return merged_df