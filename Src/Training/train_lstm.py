# src/training/train_lstm.py

from Src.Training.model_builder import (
    LSTMModelBuilder
)

from Src.Training.trainer import (
    Trainer
)


class LSTMTrainingPipeline:

    def run(

            self,

            X_train,
            y_train,

            X_val,
            y_val

    ):

        builder = LSTMModelBuilder()

        model = builder.build(

            input_steps=60,

            n_features=11,

            output_steps=60

        )

        trainer = Trainer()

        history = trainer.train(

            model,

            X_train,
            y_train,

            X_val,
            y_val
        )

        return model, history