from ctrl4ai import datasets
from ctrl4ai import prepdata
from ctrl4ai import automl
from ctrl4ai import helper
from ctrl4ai import exceptions

import pandas as pd
import numpy as np
import json

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class TestingError(Exception):
    pass


dataset1 = datasets.trip_fare()
dataset1 = dataset1.head(9999)

dataset2 = datasets.titanic()

prep = automl.Preprocessor(dataset1, learning_type='Supervised', target_variable='fare_amount', target_type='continuous')
prep.set_tranform_categorical('one_hot_encoding')
cleansed_dataset = prep.get_processed_dataset()
print(cleansed_dataset)

print(prep.get_preprocessor_artifact(file_name=r'C:\Users\SSelvaku\Documents\Temp\artifact.json'))



