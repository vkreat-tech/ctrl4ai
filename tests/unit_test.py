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


dataset = datasets.trip_fare()
dataset = dataset.head(10000)

dfs, _ = prepdata.split_dataset(dataset, n_splits=2, proportion=[0.7, 0.3], shuffle=True)

train = dfs[0]
test = dfs[1]

prep = automl.Preprocessor(train, learning_type='Supervised', target_variable='fare_amount', target_type='continuous')
prep.set_multicollinearity_check(True)
prep.set_tranform_categorical('one_hot_encoding')
cleansed_dataset = prep.get_processed_dataset()
prep.get_preprocessor_artifact(r'C:\Users\SSelvaku\Documents\Temp\artifact.json')

prep = automl.Preprocessor(train, learning_type='Supervised', target_variable='fare_amount', target_type='continuous')

artifact_file = r'C:\Users\SSelvaku\Documents\Temp\artifact.json'
artifact_json = open(artifact_file).readline()
artifact = json.loads(artifact_json)
print(artifact)








