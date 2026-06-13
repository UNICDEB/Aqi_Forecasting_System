
# src/evaluation/excel_report.py
## Code for generating and saving evaluation metrics to an Excel file.

from pathlib import Path


class ExcelReport:

    def save(

            self,

            df,

            model_name

    ):

        output_dir = Path(
            "Reports/Metrics"
        )

        output_dir.mkdir(
            exist_ok=True,
            parents=True
        )

        output_file = (

            output_dir /

            f"{model_name}_metrics.xlsx"

        )

        df.to_excel(
            output_file,
            index=False
        )

        print(
            f"Saved: {output_file}"
        )