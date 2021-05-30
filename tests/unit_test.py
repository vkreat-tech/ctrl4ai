from ctrl4ai import datasets
from ctrl4ai import prepdata
from ctrl4ai import automl
from ctrl4ai import helper

import pandas as pd

# dataset=datasets.trip_fare()
# automl.preprocess(dataset,learning_type='Supervised',target_variable='fare_amount',target_type='continuous')

# prep=automl.preprocessor(dataset,learning_type='Supervised',target_variable='fare_amount',target_type='continuous')
# print(prep.learning_type)

df = pd.read_csv(r'test_datasets/corr_nomial_continuous.csv')
print(prepdata.nominal_scale_corr(df['A'], df['B']))

a=[1,2]+[3,4]+[5,6]
print(a)
