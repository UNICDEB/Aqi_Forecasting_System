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


class Evaluator:

    def evaluate(

            self,

            model,

            X_test,

            y_test

    ):

        predictions = model.predict(
            X_test
        )

        calculator = ParameterMetrics()

        metrics_df = calculator.calculate(

            y_test,

            predictions

        )

        return predictions, metrics_df