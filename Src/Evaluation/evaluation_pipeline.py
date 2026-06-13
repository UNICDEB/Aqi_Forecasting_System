# src/evaluation/evaluation_pipeline.py
## Code for running the evaluation pipeline.

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

            scaler,

            model_name="LSTM"

    ):

        evaluator = Evaluator()

        predictions, metrics_df = evaluator.evaluate(

            model,

            X_test,

            y_test,

            scaler

        )

        report = ExcelReport()

        report.save(

            metrics_df,

            model_name

        )

        return (

            predictions,

            metrics_df

        )