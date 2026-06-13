# from Src.Evaluation.metrics import Metrics


# class Evaluator:

#     def evaluate(
#             self,
#             model,
#             X_test,
#             y_test
#     ):

#         predictions = model.predict(
#             X_test
#         )

#         results = Metrics.calculate(
#             y_test,
#             predictions
#         )

#         return (
#             predictions,
#             results
#         )

##############################

# src/evaluation/evaluator.py
## Code for Evaluator class.


from Src.Evaluation.parameter_metrics import (
    ParameterMetrics
)

from Src.Evaluation.inverse_scaler import (
    InverseScaler
)


class Evaluator:

    def evaluate(

            self,

            model,

            X_test,

            y_test,
            scaler

    ):

        predictions = model.predict(
            X_test
        )

        calculator = ParameterMetrics()


        inverse = InverseScaler()

        y_true_original = inverse.transform(
            scaler,
            y_test
        )

        y_pred_original = inverse.transform(
            scaler,
            predictions
        )

        metrics_df = calculator.calculate(
            y_true_original,
            y_pred_original
        )

        return predictions, metrics_df