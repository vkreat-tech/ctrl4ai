from ctrl4ai import datasets
from ctrl4ai import prepdata
from ctrl4ai import automl
from ctrl4ai import helper

import pandas as pd
import numpy as np


class TestingError(Exception):
    pass


# dataset=datasets.trip_fare()
# automl.preprocess(dataset,learning_type='Supervised',target_variable='fare_amount',target_type='continuous')

# prep=automl.preprocessor(dataset,learning_type='Supervised',target_variable='fare_amount',target_type='continuous')
# prep.set_categorical_threshold('')
# prep.set_categorical_threshold('')

# print(prep.learning_type)

