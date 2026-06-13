# src/feature_engineering/dataset_builder.py

from sklearn.model_selection import train_test_split


class DatasetBuilder:

    def split(
            self,
            X,
            y,
            test_size=0.15,
            val_size=0.15
    ):

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=test_size + val_size,
            shuffle=False
        )

        val_ratio = val_size / (
                test_size + val_size
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=val_ratio,
            shuffle=False
        )

        return (

            X_train,
            X_val,
            X_test,

            y_train,
            y_val,
            y_test
        )