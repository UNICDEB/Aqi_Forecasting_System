# src/training/train_gru.py

from Src.Training.gru_builder import GRUBuilder
from Src.Training.trainer import Trainer


class GRUTrainingPipeline:

    def run(
            self,
            X_train,
            y_train,
            X_val,
            y_val
    ):

        builder = GRUBuilder()

        model = builder.build(
            input_steps=X_train.shape[1],
            n_features=X_train.shape[2],
            output_steps=y_train.shape[1]
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