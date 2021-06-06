import pandas as pd
import numpy as np

from ctrl4ai import datasets
from ctrl4ai import prepdata
from ctrl4ai import automl
from ctrl4ai import helper


class TestingError(Exception):
    pass


# =============================================================================
# prepdata.nominal_scale_corr
# =============================================================================

df_dict = {
    'A': {0: 'NKJ', 1: 'NKJ', 2: 'NKJ', 3: 'NKJ', 4: 'POL', 5: 'POL', 6: 'POL', 7: 'JKL', 8: 'JKL', 9: 'JKL', 10: 'JKL',
          11: 'RTG', 12: 'RTG', 13: 'RTG'},
    'B': {0: 1, 1: 1, 2: 2, 3: 2, 4: 7, 5: 7, 6: 8, 7: 5, 8: 5, 9: 6, 10: 6, 11: 11, 12: 11, 13: 12}}
df = pd.DataFrame.from_dict(df_dict)
result = prepdata.nominal_scale_corr(df['A'], df['B'])
if result != 1:
    raise TestingError('Error in prepdata.nominal_scale_corr')

# =============================================================================


# =============================================================================
# helper.isNaN
# =============================================================================

val = np.nan
if not helper.isNaN(val):
    raise TestingError('Error in helper.isNaN')
val = 5
if helper.isNaN(val):
    raise TestingError('Error in helper.isNaN')

# =============================================================================


# =============================================================================
# prepdata.get_ordinal_encoded_df
# =============================================================================

data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'current': [True, False, True, True, False, True, True, False, True, False],
    'check': ['notatall', 'entirely', 'notatall', 'entirely', 'partially', np.nan, 'not at all', 'NOT at all',
              'notatall', 'not at all'],
    'qualify': ['Yes', 'no', 'yes', 'NO', np.NAN, 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
dataset = pd.DataFrame(data, index=labels)
labels, dataset = prepdata.get_ordinal_encoded_df(dataset)
expected_labels = {'check': {'notatall': 0, 'entirely': 2, 'partially': 1, 'not at all': 0, 'NOT at all': 0},
                   'qualify': {'Yes': 1, 'no': 0, 'yes': 1, 'NO': 0}}
if labels != expected_labels:
    raise TestingError('Error in prepdata.get_ordinal_encoded_df')

# =============================================================================


# =============================================================================
# prepdata.get_ordinal_encoded_df (with custom ordinal mapper)
# =============================================================================

data = {'level': ['level 0', 'level 1', 'level 2', 'level 3', 'level 3', 'level 2', 'level 1', 'level 0', 'level 1', 'level 2'],
        'value': [1, 6, 8, 2, 4, 5, 8, 2, 6, 8],
        'eligible': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no'],
        'grade': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', np.nan]}

df = pd.DataFrame(data)
ordinal_dict = {'level': {'level 0': 0, 'level 1': 1, 'level 2': 2, 'level 3': 3},
                'grade': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}}
labels, df = prepdata.get_ordinal_encoded_df(df, ordinal_dict)

expected_dict = {'level': {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1, 7: 0, 8: 1, 9: 2}, 'value': {0: 1, 1: 6, 2: 8, 3: 2, 4: 4, 5: 5, 6: 8, 7: 2, 8: 6, 9: 8}, 'eligible': {0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 1, 9: 0}, 'grade': {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2, 6: 0, 7: 1, 8: 2, 9: 0}}

if df.to_dict() != expected_dict:
    raise TestingError('Error in prepdata.custom_ordinal_mapper')


# =============================================================================


