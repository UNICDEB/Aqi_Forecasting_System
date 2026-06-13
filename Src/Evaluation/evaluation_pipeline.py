# src/evaluation/evaluation_pipeline.py
## Code for running the evaluation pipeline.

import pandas as pd

from Src.Evaluation.evaluator import (
    Evaluator
)

from Src.Evaluation.excel_report import (
    ExcelReport
)


class EvaluationPipeline:

    def run(

            self,

            model,

            X_test,

            y_test,

            model_name="LSTM"

    ):

        evaluator = Evaluator()

        predictions, metrics = (

            evaluator.evaluate(

                model,

                X_test,

                y_test

            )

        )

        metrics_df = metrics

        report = ExcelReport()

        report.save(

            metrics_df,

            model_name

        )

        return (

            predictions,

            metrics_df

        )