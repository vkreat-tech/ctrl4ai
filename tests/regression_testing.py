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
    raise TestingError

# =============================================================================

# =============================================================================
# helper.isNaN()
# =============================================================================

val = np.nan
if helper.isNaN(val) != True:
    raise TestingError
val = 5
if helper.isNaN(val) == True:
    raise TestingError

# =============================================================================
