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


df1 = datasets.trip_fare()
df1 = df1.head(10000)


k_datasets = prepdata.k_fold(df1, 5)

#print(k_datasets)

# train = dfs[0]
# test = dfs[1]

# prep = automl.Preprocessor(train, learning_type='Supervised', target_variable='fare_amount', target_type='continuous')
# prep.set_feature_selection(True, select_top=10)
# cleansed_dataset = prep.get_processed_dataset()
# prep.get_preprocessor_artifact(r'C:\Users\SSelvaku\Documents\Temp\artifact.json')

# print(cleansed_dataset)

# prep = automl.Preprocessor(train, learning_type='Supervised', target_variable='fare_amount', target_type='continuous')

# artifact_file = r'C:\Users\SSelvaku\Documents\Temp\artifact.json'
# artifact_json = open(artifact_file).readline()
# artifact = json.loads(artifact_json)
# print(artifact)

