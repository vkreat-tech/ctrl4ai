from ctrl4ai import datasets
from ctrl4ai import prepdata
from ctrl4ai import automl
from ctrl4ai import helper

import pandas as pd
import numpy as np

class TestingError(Exception):
    pass

#dataset=datasets.trip_fare()
#automl.preprocess(dataset,learning_type='Supervised',target_variable='fare_amount',target_type='continuous')

#prep=automl.preprocessor(dataset,learning_type='Supervised',target_variable='fare_amount',target_type='continuous')
#prep.set_categorical_threshold('')
#prep.set_categorical_threshold('')

# print(prep.learning_type)

#df = pd.read_csv(r'test_datasets/corr_nomial_continuous.csv')

#print(df.to_dict())

df_dict={'A': {0: 'NKJ', 1: 'NKJ', 2: 'NKJ', 3: 'NKJ', 4: 'POL', 5: 'POL', 6: 'POL', 7: 'JKL', 8: 'JKL', 9: 'JKL', 10: 'JKL', 11: 'RTG', 12: 'RTG', 13: 'RTG'}, 'B': {0: 1, 1: 1, 2: 2, 3: 2, 4: 7, 5: 7, 6: 8, 7: 5, 8: 5, 9: 6, 10: 6, 11: 11, 12: 11, 13: 12}}
df=pd.DataFrame.from_dict(df_dict)

print(prepdata.spearmans_corr(df['A'], df['B']))
