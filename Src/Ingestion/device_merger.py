# # src/ingestion/device_merger.py

# import pandas as pd
# from pathlib import Path


# class DeviceMerger:

#     def __init__(self):
#         pass

#     def merge_device_data(
#             self,
#             device_files
#     ):

#         all_df = []

#         for file in device_files:

#             df = pd.read_excel(
#                 file,
#                 skiprows=7
#             )

#             all_df.append(df)

#         merged_df = pd.concat(
#             all_df,
#             ignore_index=True
#         )

#         return merged_df

######################

import pandas as pd
from pathlib import Path
# Import your ExcelReader tool
from Src.Ingestion.excel_reader import ExcelReader

class DeviceMerger:

    def __init__(self):
        # Initialize the ExcelReader once when DeviceMerger is created
        self.reader = ExcelReader()

    def merge_device_data(self, device_files):
        all_df = []

        for file in device_files:
            # Use the reader instance to process each Excel file safely
            df = self.reader.read_device_file(file)
            all_df.append(df)

        # Combine all the loaded DataFrames together
        merged_df = pd.concat(all_df, ignore_index=True)

        return merged_df
