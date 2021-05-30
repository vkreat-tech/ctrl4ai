import pandas as pd

from ctrl4ai import datasets
from ctrl4ai import prepdata
from ctrl4ai import automl
from ctrl4ai import helper

# =============================================================================
# Correlation Testing - Nominal and Continuous
# =============================================================================

df = pd.read_csv(r'test_datasets/corr_nomial_continuous.csv')
print(prepdata.nominal_scale_corr(df['A'], df['B']))

# =============================================================================


