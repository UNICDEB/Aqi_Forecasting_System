# src/feature_engineering/scaler.py

import joblib

from sklearn.preprocessing import MinMaxScaler


class AQIScaler:

    def __init__(self):

        self.scaler = MinMaxScaler()

    def fit_transform(self, df):

        scaled = self.scaler.fit_transform(df)

        return scaled

    def transform(self, df):

        return self.scaler.transform(df)

    def inverse_transform(self, data):

        return self.scaler.inverse_transform(data)

    def save(self, path):

        joblib.dump(
            self.scaler,
            path
        )

    def load(self, path):

        self.scaler = joblib.load(
            path
        )