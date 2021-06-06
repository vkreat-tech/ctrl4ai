from ctrl4ai import datasets
from ctrl4ai import prepdata
from ctrl4ai import automl
from ctrl4ai import helper

import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class TestingError(Exception):
    pass


dataset = datasets.trip_fare()
dataset = dataset.head(10000)

#prep = automl.Preprocessor(dataset, learning_type='Supervised', target_variable='fare_amount', target_type='continuous')
#prep.get_processed_dataset()

data = {'level': ['level 0', 'level 1', 'level 2', 'level 3', 'level 3', 'level 2', 'level 1', 'level 0', 'level 1', 'level 2'],
        'value': [1, 6, 8, 2, 4, 5, 8, 2, 6, 8],
        'eligible': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', np.nan],
        'grade': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', np.nan]}

df = pd.DataFrame(data)
df = prepdata.get_ohe_df(df,define_nominal_cols=['level','grade'], drop_first=False)
print(df)
