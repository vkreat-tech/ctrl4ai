from ctrl4ai import datasets
from ctrl4ai import helper

import pandas as pd

help(datasets.titanic)

help(helper.one_hot_encoding)

print(helper.distance_calculator(53.32055555555556,-1.7297222222222221,53.31861111111111,-1.6997222222222223))

dataset=datasets.titanic()
print(dataset['Fare'])

print(helper.freedman_diaconis(dataset['Fare'],returnas='width'))
print(helper.freedman_diaconis(dataset['Fare'],returnas='bins'))



