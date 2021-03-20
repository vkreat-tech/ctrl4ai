from ctrl4ai import datasets
from ctrl4ai import helper
import matplotlib.pyplot as plt
import pandas as pd

dataset=datasets.titanic(refresh=True)
median_val=dataset['Age'].median()
dataset['Age']=dataset['Age'].fillna(median_val)
print(dataset['Age'].shape)
width=helper.freedman_diaconis(dataset['Age'],returnas='width')
bins=helper.freedman_diaconis(dataset['Age'],returnas='bins')
print(bins)

#dataset['Age'].hist(bins=30)
#plt.show()

dataset['Age_binned_qcut'] = pd.qcut(dataset['Age'], q=bins, duplicates='drop')
print(dataset['Age_binned_qcut'].shape)
print(dataset['Age_binned_qcut'].value_counts())

dataset['Age_binned_cut'] =pd.cut(dataset['Age'], bins=bins)
print(dataset['Age_binned_cut'].shape)
print(dataset['Age_binned_cut'].value_counts())


from datetime import datetime
now = datetime.now()
timestamp = str(datetime.timestamp(now)).replace('.','_')
print("timestamp =", timestamp)