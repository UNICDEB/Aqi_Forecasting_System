# src/training/callbacks.py

from tensorflow.keras.callbacks import (

    EarlyStopping,

    ReduceLROnPlateau,

    ModelCheckpoint

)


def get_callbacks():

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5
        ),

        ModelCheckpoint(
            filepath=
            "models/LSTM/best_model.keras",

            save_best_only=True,

            monitor="val_loss"
        )

    ]

    return callbacks