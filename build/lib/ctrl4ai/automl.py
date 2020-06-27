# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:00:36 2020

@author: ShajiJamesSelvakumar
"""

from ctrl4ai import preprocessing
from ctrl4ai import helper

def preprocess(dataset,
               learning_type,
               target_variable=None,
               impute_null_method='central_tendency',
               tranform_categorical='label_encoding',
               categorical_threshold=0.3,
               drop_null_dominated=True,
               dropna_threshold=0.7,
               derive_from_datetime=True,
               select_continuous_features='correlation',
               select_categorical_features='chi_square',
               drop_non_numeric=True,
               drop_single_valued=True
               ):
    if str.lower(learning_type)=='supervised' and target_variable==None:
        print('target_variable is a required parameter for supervised learning')
        raise helper.ParameterError
    if derive_from_datetime:
        dataset=preprocessing.derive_from_datetime(dataset)
    return dataset

