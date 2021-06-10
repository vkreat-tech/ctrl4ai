# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:27:31 2020

@author: Shaji,Charu,Selva
"""

import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from scipy import stats

from . import _ordinal_dictionary

pd.set_option('mode.chained_assignment', None)


def isNaN(num):
    """
    Usage: [arg1]:[numeric value]
    Description: Checks if the value is null (numpy.NaN)
    Returns: Boolean [True/False]
    """
    return num != num


def added_constant_log(dataset,
                       col,
                       min_value=None):
    """
    Usage: [arg1]:[dataset], [arg2]:[column in which log transform should be done]
    Description: Log transforms the specified column
    Returns: DataFrame
    """
    if min_value is None:
        min_value = dataset[col].min()
    if min_value <= 0:
        dataset[col] = dataset[col].apply(lambda x: np.log(x + np.abs(min_value) + 1))
    else:
        dataset[col] = dataset[col].apply(lambda x: np.log(x))
    return dataset


def yeojohnsonlog(x):
    """
    Usage: [arg1]:[real/float value]
    Description: Log transforms the specified column based on Yeo Joshson Power Transform
    Returns: Log value (numeric)
    """
    if x < 0:
        y = -np.log(-x + 1)
    else:
        y = np.log(x + 1)
    return y


def distance_calculator(start_latitude,
                        start_longitude,
                        end_latitude,
                        end_longitude):
    """
    Usage: [arg1]:[numeric-start_latitude],[arg2]:[numeric-start_longitude],[arg3]:[numeric-end_latitude],[arg4]:[numeric-end_longitude]
    Returns: Numeric [Distance in kilometers]
    """
    if isNaN(start_latitude) or isNaN(start_longitude) or isNaN(end_latitude) or isNaN(end_longitude):
        return np.NaN
    else:
        lat1 = radians(start_latitude)
        lon1 = radians(start_longitude)
        lat2 = radians(end_latitude)
        lon2 = radians(end_longitude)
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        # Radius of earth in kilometers.
        r = 6371
        return (c * r)


def test_numeric(test_string):
    """
    Usage: [arg1]:[String/Number]
    Description: Checks if the value is numeric
    Returns: Boolean [True/False]
    """
    try:
        float(test_string)
        res = True
    except:
        res = False
    return res


def check_numeric_col(col_series):
    """
    Usage: [arg1]:[Pandas Series / Single selected column of a dataframe]
    Description: Checks if all the values in the series are numerical
    Returns: Boolean [True/False]
    """
    if all(col_series.apply(lambda x: test_numeric(x))):
        return True
    else:
        return False


def check_ordinal_col(col_series):
    """
    Usage: [arg1]:[Pandas Series / Single selected column of a dataframe]
    Description: Checks if all the column contains ordinal values checking against the Ctrl4AI's data dictionary
    Returns: Boolean [True/False], dict [ordinal to numeric mapper]
    """
    result = False
    result_dict = dict()
    if not check_numeric_col(col_series):
        mode_val = col_series.mode()[0]
        col_series = col_series.fillna(mode_val).astype('str')
        distinct_elements = list(col_series.unique())
        unique_elements = [str.lower(val.replace(' ', '')) for val in col_series.unique()]
        unique_elements = list(set(unique_elements))
        count = len(unique_elements)
        possible_scales = _ordinal_dictionary._get_possible_scales(count)
        for scale in possible_scales:
            unique_keys = [str.lower(val.replace(' ', '')) for val in scale.keys()]
            if set(unique_keys) == set(unique_elements):
                result = True
                transformed_mapper = dict()
                for key in scale.keys():
                    new_key = str.lower(key.replace(' ', ''))
                    transformed_mapper[new_key] = scale[key]
                for val in distinct_elements:
                    result_dict[val] = transformed_mapper[str.lower(val.replace(' ', ''))]
    return result, result_dict


def check_categorical_col(col_series,
                          categorical_threshold=0.3):
    """
    Usage: [arg1]:[Pandas Series / Single selected column of a dataframe],[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
    Description: Breaks the values to chunks and checks if the proportion of unique values is less than the threshold
    Returns: Boolean [True/False]
    """
    col_array = np.array(col_series.apply(lambda x: str(x).strip()).astype(str).fillna(str(0)))
    if col_array.size >= 1000:
        n = 10
        k = 100
    elif col_array.size > 100:
        n = np.floor(col_array.size / 100)
        k = 100
    else:
        n = 1
        k = col_array.size
    if n % 2 == 0:
        n -= 1
    t = 0
    f = 0
    for i in range(int(n)):
        sample = np.random.choice(col_array, size=k, replace=False)
        if np.unique(sample).size / sample.size <= categorical_threshold:
            t += 1
        else:
            f += 1
    if t > f:
        return True
    else:
        return False


def single_valued_col(col_series):
    """
    Usage: [arg1]:[Pandas Series / Single selected column of a dataframe]
    Description: Checks if the column has only one value
    Returns: Boolean [True/False]
    """
    if col_series.dropna().unique().shape[0] == 1:
        return True
    else:
        return False


def one_hot_encoding(dataset,
                     categorical_cols_list,
                     drop_first=True):
    """
    Usage: [arg1]:[pandas dataframe],[arg2]:[list of columns to be encoded]
    Description: Transformation for categorical features by getting dummies
    Returns: Dataframe [with separate column for each categorical values]
    """
    dataset = pd.merge(dataset, pd.get_dummies(dataset[categorical_cols_list], columns=categorical_cols_list,
                                               drop_first=drop_first),
                       left_index=True, right_index=True)
    dataset = dataset.drop(categorical_cols_list, axis=1)
    return dataset


def freedman_diaconis(data, returnas="width"):
    """
    Usage: [arg1]:[Pandas Series],[arg2]:[returnas: {"width", "bins"}]
    Description: Use Freedman Diaconis rule to compute optimal histogram bin width. ``returnas`` can be one of "width" or "bins", indicating whether the bin width or number of bins should be returned respectively.
    Returns: Numeric [Width/No.of bins - whatever is opted]
    """
    data = np.asarray(data, dtype=np.float_)
    IQR = stats.iqr(data, rng=(25, 75), scale=1.0, nan_policy="omit")
    N = data.size
    bw = (2 * IQR) / np.power(N, 1 / 3)

    if returnas == "width":
        result = bw
    else:
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int((datrng / bw) + 1)
    return result


def bool_to_int(dataset):
    """
    Usage: [arg1]:[pandas dataframe]
    Description: Transformation for boolean features to integers
    Returns: Dataframe [with booleans converted to integers]
    """
    for col in dataset:
        if check_categorical_col(dataset[col]):
            mode_val = dataset[col].mode()[0]
            dataset[col] = dataset[col].fillna(mode_val)
            if dataset[col].dtype == 'bool':
                dataset[col] = dataset[col].astype('int')
    return dataset


def get_columns_subsets(cols, min_cols=1):
    """
    Usage: [arg1]:[list of columns], [min_cols (default=1):[values in the range of 1 to length of columns]
    Description: Gets all subsets of the column list
    Returns: [column list]
    """
    lists = []
    col_list = list(cols)
    for i in range(len(col_list) + 1):
        for j in range(i):
            subset = col_list[j: i]
            subset.sort()
            if len(subset) >= min_cols:
                lists.append(subset)
    return lists


def get_absolute(num):
    """
    Usage: [arg1]:[numeric value]
    Description: Converts to a positive number
    Returns: [positive numeric value]
    """
    if num >= 0:
        return num
    else:
        return -num


def correlation_threshold(rows):
    return 2 / np.sqrt(rows)


def collinearity_threshold(rows):
    if rows <= 100:
        return 0.99
    else:
        return 2 / np.log10(rows)


def intersection(seq1, seq2):
    seq3 = [value for value in seq1 if value in seq2]
    return seq3

