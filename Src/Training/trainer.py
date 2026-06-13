# # src/training/trainer.py

# class Trainer:

#     def train(
#             self,
#             model,

#             X_train,
#             y_train,

#             X_val,
#             y_val
#     ):

#         history = model.fit(

#             X_train,

#             y_train,

#             validation_data=(
#                 X_val,
#                 y_val
#             ),

#             epochs=50,

#             batch_size=64,

#             callbacks=None,

#             verbose=1
#         )

#         return history

########################

# src/training/trainer.py

# Import your custom callbacks function
from Src.Training.callbacks import get_callbacks


class Trainer:

    def train(self, model, X_train, y_train, X_val, y_val):

        # Fetch the initialized callbacks array dynamically
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=64,
            callbacks=get_callbacks(),
            verbose=1,
        )

        return history
