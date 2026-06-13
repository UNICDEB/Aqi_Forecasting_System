import pandas as pd

class Validator:

    def validate(self, df):

        report = {}

        report["missing"] = df.isnull().sum()

        report["duplicates"] = df.duplicated().sum()

        report["shape"] = df.shape

        return report