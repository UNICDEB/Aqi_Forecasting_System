from Src.Evaluation.metrics import Metrics


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

        results = Metrics.calculate(
            y_test,
            predictions
        )

        return (
            predictions,
            results
        )