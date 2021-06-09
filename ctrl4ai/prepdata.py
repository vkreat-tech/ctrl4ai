# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:54:12 2020

@author: Shaji,Charu,Selva
"""

import scipy
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from . import helper
from . import exceptions

pd.set_option('mode.chained_assignment', None)


def get_distance(dataset,
                 start_latitude,
                 start_longitude,
                 end_latitude,
                 end_longitude):
    """
    Usage: [arg1]:[Pandas DataFrame],[arg2]:[column-start_latitude],[arg3]:[column-start_longitude],[arg4]:[column-end_latitude],[arg5]:[column-end_longitude]
    Returns: DataFrame with additional column [Distance in kilometers]
    """
    print(
        "This module (ctrl4ai.preprocessing) will be deprecated by the end of 2021. Please plan to switch to the same functions in ")
    dataset['kms_' + start_latitude + '_' + end_latitude] = dataset.apply(
        lambda row: helper.distance_calculator(row[start_latitude], row[start_longitude], row[end_latitude],
                                               row[end_longitude]), axis=1)
    return dataset


def get_timediff(dataset,
                 start_time,
                 end_time):
    """
    Usage: [arg1]:[Pandas DataFrame],[arg2]:[column-start_time],[arg3]:[column-end_time]
    Returns: DataFrame with additional column [Duration in seconds]
    """
    dataset['secs_diff_' + start_time + '_' + end_time] = (dataset[end_time] - dataset[start_time]).dt.total_seconds()
    return dataset


def derive_from_datetime(dataset):
    """
    Usage: [arg1]:[pandas dataframe]
    Prerequisite: Type for datetime columns to be defined correctly
    Description: Derives the hour, weekday, year and month from a datetime column
    Returns: Dataframe [with new columns derived from datetime columns]
    """
    columns = []
    for column, dtype in dataset.dtypes.items():
        if 'datetime' in str(dtype):
            columns.append(column)
            dataset['hour_of_' + column] = dataset[column].apply(lambda x: x.hour)
            dataset['weekday_of_' + column] = dataset[column].apply(lambda x: x.weekday())
            dataset['year_of_' + column] = dataset[column].apply(lambda x: x.year)
            dataset['month_of_' + column] = dataset[column].apply(lambda x: x.month)
    return dataset, columns


def log_transform(dataset, method='yeojohnson', define_continuous_cols=[], ignore_cols=[], categorical_threshold=0.3):
    """
    Usage: [arg1]:[pandas dataframe],[method]=['yeojohnson'/'added_constant']
    Description: Checks if the a continuous column is skewed and does log transformation
    Returns: Dataframe [with all skewed columns normalized using appropriate approach]
    """
    continuous_columns = []
    for col in define_continuous_cols:
        if col not in ignore_cols:
            continuous_columns.append(col)

    for col in dataset.columns:
        if col not in continuous_columns+ignore_cols:
            if helper.check_categorical_col(dataset[col], categorical_threshold=categorical_threshold) == False and helper.check_numeric_col(dataset[col]):
                continuous_columns.append(col)

    for col in dataset.columns:
        if col in continuous_columns and np.abs(scipy.stats.skew(dataset[col])) > 1:
            print('Log Normalization(' + method + ') applied for ' + col)
            if method == 'yeojohnson':
                dataset[col] = dataset[col].apply(lambda x: helper.yeojohnsonlog(x))
            elif method == 'added_constant':
                dataset = helper.added_constant_log(dataset, col)
    return dataset


def drop_null_fields(dataset,
                     dropna_threshold=0.7, ignore_cols=[]):
    """
    Usage: [arg1]:[pandas dataframe],[dropna_threshold(default=0.7)]:[What percentage of nulls should account for the column top be removed],[ignore_cols]:[columnd that shouldn't be dropped]
    Description: Drop columns that has more null values
    Returns: Dataframe [with null dominated columns removed]
    """
    no_of_records = dataset.shape[0]
    select_cols = []
    dropped_cols = []
    for index, val in dataset.isnull().sum().items():
        if val / no_of_records < dropna_threshold or index in ignore_cols:
            select_cols.append(index)
        else:
            dropped_cols.append(index)
            print('Dropping null dominated column(s) ' + index)
    return dataset[select_cols], dropped_cols


def drop_single_valued_cols(dataset):
    """
    Usage: [arg1]:[pandas dataframe]
    Description: Drop columns that has only one value in it
    Returns: Dataframe [without single valued columns]
    """
    single_valued_cols = []
    for col in dataset.columns:
        if helper.single_valued_col(dataset[col]):
            single_valued_cols.append(col)
    if len(single_valued_cols) > 0:
        print('Dropping single valued column(s) ' + ','.join(single_valued_cols))
        dataset = dataset.drop(single_valued_cols, axis=1)
    return dataset


def get_ohe_df(dataset,
               target_variable=None,
               define_nominal_cols=[],
               ignore_cols=[],
               drop_first=True,
               categorical_threshold=0.3):
    """
    Usage: [arg1]:[pandas dataframe],[target_variable(default=None)]:[Dependent variable for Regression/Classification],[ignore_cols]:[categorical columns where one hot encoding need not be done],[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
    Description: Auto identifies categorical features in the dataframe and does one hot encoding
    Note: Consumes more system mermory if the size of the dataset is huge
    Returns: Dataframe [with separate column for each categorical values]
    """
    nominal_cols = []
    nominal_cols.extend(define_nominal_cols)
    columns = []
    for col in dataset:
        if col not in nominal_cols+ignore_cols and col != target_variable:
            if helper.check_categorical_col(dataset[col], categorical_threshold=categorical_threshold):
                nominal_cols.append(col)
    for col in dataset.columns:
        if col in nominal_cols and col not in ignore_cols:
            print('One hot encoding ' + col)
            columns.append(col)
            dataset = helper.one_hot_encoding(dataset, [col], drop_first=drop_first)
    return dataset, columns


def drop_non_numeric(dataset):
    """
    Usage: [arg1]:[pandas dataframe]
    Description: Drop columns that are not numeric
    Returns: Dataframe [only numeric features]
    """
    drop_cols = []
    for col in dataset.columns:
        if helper.check_numeric_col(dataset[col]) == False:
            drop_cols.append(col)
    if len(drop_cols) > 0:
        print("Dropping non categorical/continuous column(s):" + ','.join(drop_cols))
        dataset = dataset.drop(drop_cols, axis=1)
    return dataset


def impute_nulls(dataset,
                 method='central_tendency',
                 define_continuous_cols=[],
                 define_nominal_cols=[],
                 define_ordinal_cols=[],
                 categorical_threshold=0.3):
    """
    Usage: [arg1]:[pandas dataframe],[method(default=central_tendency)]:[Choose either central_tendency or KNN]
    Description: Auto identifies the type of distribution in the column and imputes null values
    Note: KNN consumes more system memory if the size of the dataset is huge
    Returns: Dataframe [with separate column for each categorical values]
    """
    nominal_cols = []
    ordinal_cols = []
    continuous_cols = []
    nominal_cols.extend(define_nominal_cols)
    continuous_cols.extend(define_continuous_cols)
    ordinal_cols.extend(define_ordinal_cols)
    for col in dataset:
        if col not in nominal_cols + ordinal_cols + continuous_cols:
            if helper.check_categorical_col(dataset[col], categorical_threshold=categorical_threshold):
                nominal_cols.append(col)
            elif helper.check_numeric_col(dataset[col]):
                continuous_cols.append(col)
    if str.lower(method) == 'knn':
        for col, value in dataset.isnull().sum().items():
            if value > 0:
                if col in nominal_cols + ordinal_cols:
                    print("KNN (Only Categorical): Replaced nulls in " + col + " with mode")
                    mode_val = dataset[col].mode()[0]
                    dataset[col] = dataset[col].fillna(mode_val)
        k_knn = int(np.ceil(np.sqrt(dataset.shape[0])))
        if k_knn % 2 == 0:
            k_knn += 1
        imputer = KNNImputer(n_neighbors=k_knn)
        knn_imputed_array = imputer.fit_transform(dataset)
        dataset = pd.DataFrame(knn_imputed_array, columns=dataset.columns)
        return dataset
    elif method == 'central_tendency':
        for col, value in dataset.isnull().sum().items():
            if value > 0:
                if col in nominal_cols + ordinal_cols:
                    print("Replaced nulls in " + col + " with mode")
                    mode_val = dataset[col].mode()[0]
                    dataset[col] = dataset[col].fillna(mode_val)
                elif col in continuous_cols:
                    if np.abs(scipy.stats.skew(dataset[col])) > 1:
                        print("Replaced nulls in " + col + " with median")
                        median_val = dataset[col].median()
                        dataset[col] = dataset[col].fillna(median_val)
                    else:
                        print("Replaced nulls in " + col + " with mean")
                        mean_val = dataset[col].mean()
                        dataset[col] = dataset[col].fillna(mean_val)
        return dataset
    else:
        raise exceptions.ParameterError('Method should be either central_tendency or knn')


def label_encode(dataset,
                 col):
    """
    Usage: [arg1]:[pandas dataframe],[arg1]:[column to be encoded]
    Description: Labelling categorical features with numbers from 0 to n categories
    Returns: Label Dict , Dataframe
    """
    mode_val = dataset[col].mode()[0]
    # dataset[col] = dataset[col].apply(lambda x: str(x).strip()).astype(str).fillna(mode_val)
    dataset[col] = dataset[col].fillna(mode_val)
    label_dict = dict(zip(dataset[col].unique(), np.arange(dataset[col].unique().shape[0])))
    dataset = dataset.replace({col: label_dict})
    dataset[col] = dataset[col].astype('int')
    dataset[col] = dataset[col].astype('category')
    return label_dict, dataset


def remove_outlier_df(dataset,
                      cols):
    """
    Usage: [arg1]:[pandas dataframe],[arg2]:[list of columns to check and remove outliers]
    Description: The column needs to be continuous
    Returns: DataFrame with outliers removed for the specific columns
    """
    for col in cols:
        outlier_temp_dataset = pd.DataFrame(dataset[col])
        outlier_temp_dataset = impute_nulls(outlier_temp_dataset)
        Q1 = outlier_temp_dataset.quantile(0.25)
        Q3 = outlier_temp_dataset.quantile(0.75)
        IQR = Q3 - Q1
        outlier_bool_dataset = ((outlier_temp_dataset > (Q1 - 1.5 * IQR)) & (outlier_temp_dataset < (Q3 + 1.5 * IQR)))
        select_index = outlier_bool_dataset.index[outlier_bool_dataset[col] == True]
        print('No. of outlier rows removed based on ' + col + ' is ' + str(
            outlier_temp_dataset.shape[0] - len(select_index)))
        dataset = dataset.iloc[select_index].reset_index(drop=True)
    return dataset


def auto_remove_outliers(dataset,
                         ignore_cols=[],
                         categorical_threshold=0.3,
                         define_continuous_cols=[]):
    """
    Usage: [arg1]:[pandas dataframe],[ignore_cols]:[list of columns to be ignored],[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
    Description: Checks if the column is continuous and removes outliers
    Returns: DataFrame with outliers removed
    """
    continuous_columns = []
    for col in define_continuous_cols:
        if col not in ignore_cols:
            continuous_columns.append(col)
    for col in dataset.columns:
        if col not in continuous_columns+ignore_cols:
            if helper.check_categorical_col(dataset[col], categorical_threshold=categorical_threshold) == False and helper.check_numeric_col(dataset[col]) == True:
                continuous_columns.append(col)
    dataset = remove_outlier_df(dataset, continuous_columns)
    return dataset


def get_label_encoded_df(dataset,
                         categorical_threshold=0.3,
                         define_nominal_cols=[],
                         ignore_cols=[]):
    """
    Usage: [arg1]:[pandas dataframe],[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
    Description: Auto identifies categorical features in the dataframe and does label encoding
    Returns: Dictionary [Labels for columns],Dataframe [with separate column for each categorical values]
    """
    nominal_cols = []
    nominal_cols.extend(define_nominal_cols)
    for col in dataset:
        if col not in nominal_cols+ignore_cols:
            if helper.check_categorical_col(dataset[col], categorical_threshold=categorical_threshold):
                nominal_cols.append(col)
    column_labels = dict()
    for col in dataset.columns:
        if col not in ignore_cols:
            if helper.check_numeric_col(dataset[col]):
                continue
            elif col in nominal_cols:
                labels, dataset = label_encode(dataset, col)
                print('Labels for ' + col + ': ' + str(labels))
                column_labels[col] = labels
    return column_labels, dataset


def get_ordinal_encoded_df(dataset, custom_ordinal_dict=dict()):
    """
    Usage: [arg1]:[pandas dataframe],[arg2]:[Pre-defined ordinal scale dictionary]
    Description: Identifies ordinal columns and translate them to numbers
    Returns: Dictionary [Labels for columns], Dataframe [with ordinal values converted to number]
    """
    column_labels = dict()
    for col in dataset:
        if col in custom_ordinal_dict.keys():
            dataset[col] = dataset[col].astype(str).map(custom_ordinal_dict[col])
            mode_val = dataset[col].mode()[0]
            dataset[col] = dataset[col].fillna(mode_val).astype('int')
        else:
            result, mapper = helper.check_ordinal_col(dataset[col])
            if result:
                dataset[col] = dataset[col].astype(str).map(mapper)
                mode_val = dataset[col].mode()[0]
                dataset[col] = dataset[col].fillna(mode_val).astype('int')
                column_labels[col] = mapper
                print('Labels for ' + col + ': ' + str(mapper))
    return column_labels, dataset


def cramersv_corr(x, y):
    """
    Usage: [arg1]:[categorical series],[arg2]:[categorical series]
    Description: Cramer's V Correlation is a measure of association between two categorical variables
    Returns: A value between 0 and +1
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def kendalltau_corr(x, y):
    """
    Usage: [arg1]:[continuous series],[arg2]:[categorical series]
    Description: Kendall Tau Correlation is a measure of association between a ordinal feature and a ordinal feature
    Returns: A value between -1 and +1
    """
    x_arr = np.array(impute_nulls(pd.DataFrame(x)))
    y_arr = np.array(impute_nulls(pd.DataFrame(y)))
    corr, _ = scipy.stats.kendalltau(x_arr, y_arr)
    return corr


def spearmans_corr(x, y):
    """
    Usage: [arg1]:[continuous series],[arg2]:[categorical series]
    Description: Spearman Correlation is a measure of association between a continuous feature and a ordinal/continuous feature with monotonic relationship
    Returns: A value between -1 and +1
    """
    x_arr = np.array(impute_nulls(pd.DataFrame(x)))
    y_arr = np.array(impute_nulls(pd.DataFrame(y)))
    corr, _ = scipy.stats.spearmanr(x_arr, y_arr)
    return corr


def pearson_corr(x, y):
    """
    Usage: [arg1]:[continuous series],[arg2]:[continuous series]
    Description: Pearson Correlation is a measure of association between two continuous features
    Returns: A value between -1 and +1
    """
    x = pd.to_numeric(x)
    y = pd.to_numeric(y)
    return np.corrcoef(x, y)[0, 1]


def nominal_scale_corr(nominal_series, continuous_series):
    """
    Usage: [arg1]:[nominal series],[arg2]:[continuous series]
    Description: Ctrl4AI's Nominal Scale Correlation is a measure of association between a nominal feature and a continuous feature
    Returns: A value between 0 and 1
    """
    mean_val = continuous_series.mean()
    continuous_series = continuous_series.fillna(mean_val)
    len_nominal = len(nominal_series.unique())
    best_corr = 0
    for bin_size in ['even', 'distributed']:
        for bins in [None, len_nominal]:
            binned_series = binning(continuous_series, bin_size=bin_size, bins=bins)
            corr_val = cramersv_corr(nominal_series, binned_series)
            if corr_val > best_corr:
                best_corr = corr_val
    return best_corr


def get_correlated_features(dataset,
                            target_col,
                            target_type,
                            correlation_threshold=None,
                            categorical_threshold=0.3,
                            define_continuous_cols=[],
                            define_nominal_cols=[],
                            define_ordinal_cols=[]):
    """
    Usage: [arg1]:[pandas dataframe],[arg2]:[target/dependent variable],[arg3]:['continuous'/'categorical'],[correlation_threshold(default=2/sqrt(dataset.shape[0]))]:[The threshold value for a good correlation],[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
    Description: Only for supervised learning to select independent variables that has some correlation with target/dependent variable (Uses Pearson correlation between two continuous variables, CramersV correlation between two categorical variables, Kendalls Tau correlation between a categorical and a continuos variable)
    Returns: Dictionary of correlation coefficients, List of columns that have considerable correlation
    """
    nominal_cols = []
    ordinal_cols = []
    continuous_cols = []
    nominal_cols.extend(define_nominal_cols)
    ordinal_cols.extend(define_ordinal_cols)
    continuous_cols.extend(define_continuous_cols)
    col_corr = dict()
    if correlation_threshold is None:
        correlation_threshold = 2 / np.sqrt(dataset.shape[0])
    for col in dataset:
        if col not in nominal_cols + continuous_cols + ordinal_cols:
            if col != target_col:
                if helper.check_categorical_col(dataset[col], categorical_threshold=categorical_threshold):
                    nominal_cols.append(col)
                elif helper.check_numeric_col(dataset[col]):
                    continuous_cols.append(col)
    if target_type == 'continuous':
        for col in continuous_cols:
            coeff = pearson_corr(dataset[col], dataset[target_col])
            col_corr[col] = coeff
        for col in ordinal_cols:
            coeff = kendalltau_corr(dataset[col], dataset[target_col])
            col_corr[col] = coeff
        for col in nominal_cols:
            coeff = nominal_scale_corr(dataset[col], dataset[target_col])
            col_corr[col] = coeff
    if target_type == 'categorical':
        for col in continuous_cols:
            coeff = kendalltau_corr(dataset[col], dataset[target_col])
            col_corr[col] = coeff
        for col in ordinal_cols + nominal_cols:
            coeff = cramersv_corr(dataset[col], dataset[target_col])
            col_corr[col] = coeff
    selected_features = []
    for col in col_corr.keys():
        if np.abs(float(col_corr[col])) > np.abs(correlation_threshold):
            selected_features.append(col)
    return col_corr, selected_features


def binning(pdSeries, bin_size='even', bins=None):
    """
  Usage: [arg1]:[Pandas Series],[bin_size(default=even)]:[even/distributed]
  Description: Will split to intervals of equal size of bin size is even. Otherwise, data will be distributed to variable bin sizes with more or less same frequency of data
  Returns: Pandas Series with Values converted to Intervals
  """
    if bins is None:
        bins = helper.freedman_diaconis(pdSeries, returnas='bins')
    if str.lower(bin_size) == 'even':
        new_pdSeries = pd.cut(pdSeries, bins=bins)
    else:
        new_pdSeries = pd.qcut(pdSeries, q=bins, duplicates='drop')
    return new_pdSeries


def multicollinearity_check(corr_df, threshold=0.7):
    """
    Usage: [arg1]:[Correlation Result DataFrame],[threshold(default=0.7)]:[Value in the range of 0-1]
    Description: Will split to intervals of equal size of bin size is even. Otherwise, data will be distributed to variable bin sizes with more or less same frequency of data
    Returns: Pandas Series with Values converted to Intervals
    """
    result_set = []
    for col in corr_df.columns:
        for row in corr_df[col].index:
            if col != row:
                val = corr_df[col][row]
                if helper.get_absolute(val) >= threshold:
                    cols = [col, row]
                    cols.sort()
                    if (cols, val) not in result_set:
                        result_set.append((cols, val))
    return result_set


def dataset_summary(dataset,
                    define_continuous_cols=[],
                    define_nominal_cols=[],
                    define_ordinal_cols=[],
                    categorical_threshold=0.3):
    """
    Usage: [arg1]:[pandas dataframe]
    Description: Returns summary of DataFrame
    Returns: [Summary Dict]
    """
    nominal_cols = []
    ordinal_cols = []
    continuous_cols = []
    nominal_cols.extend(define_nominal_cols)
    continuous_cols.extend(define_continuous_cols)
    ordinal_cols.extend(define_ordinal_cols)
    for col in dataset:
        if col not in nominal_cols + ordinal_cols + continuous_cols:
            if helper.check_categorical_col(dataset[col], categorical_threshold=categorical_threshold):
                nominal_cols.append(col)
            elif helper.check_numeric_col(dataset[col]):
                continuous_cols.append(col)
    dataset_summary = dict()
    for col in ordinal_cols:
        dataset_summary[col] = dict()
        dataset_summary[col]['type'] = 'ordinal'
        col_summary = dataset[col].describe().to_dict()
        dataset_summary[col].update(col_summary)
        dataset_summary[col]['mode'] = dataset[col].mode()[0]
        dataset_summary[col]['min'] = dataset[col].min()
        dataset_summary[col]['max'] = dataset[col].max()
    for col in nominal_cols:
        dataset_summary[col] = dict()
        dataset_summary[col]['type'] = 'nominal'
        col_summary = dataset[col].describe().to_dict()
        dataset_summary[col].update(col_summary)
        dataset_summary[col]['mode'] = dataset[col].mode()[0]
    for col in continuous_cols:
        dataset_summary[col] = dict()
        dataset_summary[col]['type'] = 'continuous'
        col_summary = dataset[col].describe().to_dict()
        dataset_summary[col].update(col_summary)
        dataset_summary[col]['mean'] = dataset[col].mean()
        dataset_summary[col]['median'] = dataset[col].median()
        dataset_summary[col]['min'] = dataset[col].min()
        dataset_summary[col]['max'] = dataset[col].max()
        if np.abs(scipy.stats.skew(dataset[col])) > 1:
            dataset_summary[col]['Skewed'] = 'Y'
        else:
            dataset_summary[col]['Skewed'] = 'N'
    return dataset_summary


def split_dataset(dataset, n_splits, proportion=None, mode=None, shuffle=False):
    if mode == 'equal':
        each_proportion = int((1/n_splits)*100)
        proportion= [each_proportion for i in range(n_splits-1)]
        final_val = 100 - sum(proportion)
        proportion.append(final_val)
        proportion = [val/100 for val in proportion]
    if len(proportion) != n_splits:
        raise exceptions.ParameterError('n_splits should be equal to the number of values in proportion')
    if sum(proportion) != 1:
        raise exceptions.ParameterError('The sum of values in proportion should be 1')

    indices = list(dataset.index)

    if shuffle:
        np.random.shuffle(indices)

    df_list = []
    indices_split = []
    prev = -1
    length = len(indices)
    for ctr in range(n_splits):
        max_records = int(np.floor(proportion[ctr] * length))
        start = prev + 1
        end = start + max_records
        curr_split = indices[start:end+1]
        if n_splits-ctr == 1:
            curr_split = indices[start:]
        indices_split.append(curr_split)
        curr_df = dataset.iloc[curr_split]
        curr_df = curr_df.reset_index()
        df_list.append(curr_df)
        prev = end
    return df_list, indices_split


def Xy_split(dataset, target_feature):
    X = dataset.drop([target_feature], axis=1)
    y = dataset[[target_feature]]
    return X, y

