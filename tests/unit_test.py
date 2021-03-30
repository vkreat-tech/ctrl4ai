from ctrl4ai import datasets
from ctrl4ai import prepdata
from ctrl4ai import automl
from ctrl4ai import helper

#dataset=datasets.trip_fare()
#automl.preprocess(dataset,learning_type='Supervised',target_variable='fare_amount',target_type='continuous')


import pandas as pd
import numpy as np
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
              'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
              'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
              'current': [True,False,True,True,False,True,True,False,True,False],
              'check': ['notatall', 'entirely', 'notatall', 'entirely', 'partially', np.nan, 'not at all', 'NOT at all', 'notatall', 'not at all'],
              'qualify': ['Yes', 'no', 'yes', 'NO', np.NAN, 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
dataset = pd.DataFrame(exam_data , index=labels)
print(dataset)

labels,dataset=prepdata.get_ordinal_encoded_df(dataset)
print(dataset)
print('\n')
print(labels)
