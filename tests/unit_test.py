from ctrl4ai import datasets
from ctrl4ai import prepdata
from ctrl4ai import automl
from ctrl4ai import helper
import pandas as pd

data = pd.DataFrame([[True, False, True], [False, False, True]])
print(data)
converted=helper.bool_to_int(data)
print(converted)
