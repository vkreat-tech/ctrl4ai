# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:00:36 2020

@author: Shaji,Charu,Selva
"""

from . import prepdata
from . import helper
from . import exceptions

import sklearn
import pandas as pd
import numpy as np
from pickle import dump
from datetime import datetime
from itertools import combinations

pd.set_option('mode.chained_assignment', None)


def preprocess(dataset,
               learning_type,
               target_variable=None,
               target_type=None,
               impute_null_method='central_tendency',
               tranform_categorical='label_encoding',
               categorical_threshold=0.3,
               remove_outliers=False,
               log_transform=None,
               drop_null_dominated=True,
               dropna_threshold=0.7,
               derive_from_datetime=True,
               ohe_ignore_cols=[],
               feature_selection=True,
               define_continuous_cols=[],
               define_categorical_cols=[]
               ):
    """
    dataset=pandas DataFrame (required)
    learning_type='supervised'/'unsupervised' (required)
    target_variable=Target/Dependent variable (required for supervised learning type)
    target_type='continuous'/'categorical' (required for supervised learning type)
    impute_null_method='central_tendency' (optional) [Choose between 'central_tendency' and 'KNN']
    tranform_categorical='label_encoding' (optional) [Choose between 'label_encoding' and 'one_hot_encoding']
    categorical_threshold=0.3 (optional) [Threshold for determining categorical column based on the percentage of unique values]
    remove_outliers=False (optional) [Choose between True and False]
    log_transform=None (optional) [Choose between 'yeojohnson'/'added_constant']
    drop_null_dominated=True (optional) [Choose between True and False - Optionally change threshold in dropna_threshold if True]
    dropna_threshold=0.7 (optional) [Proportion check for dropping dull dominated column]
    derive_from_datetime=True (optional) [derive hour, year, month, weekday etc from datetime column - make sure that the dtype is datetime for the column]
    ohe_ignore_cols=[] (optional) [List - if tranform_categorical=one_hot_encoding, ignore columns not to be one hot encoded]
    feature_selection=True (optional) [Choose between True and False - Uses Pearson correlation between two continuous variables, CramersV correlation between two categorical variables, Kendalls Tau correlation between a categorical and a continuos variable]
    define_continuous_cols=[] (optional) [List - Predefine continuous variables]
    define_categorical_cols=[] (optional) [List - Predefine categorical variables]
    |
    |
    returns [Dict - Label Encoded Columns and Values], [DataFrame - Processed Dataset]
    """

    col_labels = dict()
    if str.lower(learning_type) not in ['supervised', 'unsupervised']:
        raise exceptions.ParameterError('learning_type should be supervised/unsupervised')
    if str.lower(tranform_categorical) not in ['label_encoding', 'one_hot_encoding']:
        raise exceptions.ParameterError('learning_type should be label_encoding/one_hot_encoding')
    if str.lower(learning_type) == 'supervised' and target_variable is None:
        raise exceptions.ParameterError('target_variable is a required parameter for supervised learning')
    if str.lower(learning_type) == 'supervised' and target_type is None:
        raise exceptions.ParameterError(
            'target_type (continuous/categorical) is a required parameter for supervised learning')

    # resetting the index of the dataset
    dataset = dataset.reset_index(drop=True)
    if derive_from_datetime:
        dataset = prepdata.derive_from_datetime(dataset)

    # remove null dominated fields based on threshold if the flag is true
    if drop_null_dominated:
        dataset = prepdata.drop_null_fields(dataset, dropna_threshold=dropna_threshold)

    dataset = helper.bool_to_int(dataset)

    # drop all single valued columns
    dataset = prepdata.drop_single_valued_cols(dataset)

    # transform ordinal columns to integer values
    ordinal_labels, dataset = prepdata.get_ordinal_encoded_df(dataset)
    col_labels.update(ordinal_labels)
    ordinal_cols = [col for col in ordinal_labels.keys() if col != target_variable]
    print('Columns identified as ordinal are ' + ','.join(ordinal_cols))
    reserved_cols = []
    reserved_cols.extend(define_continuous_cols)
    reserved_cols.extend(define_categorical_cols)
    reserved_cols.extend(ordinal_cols)

    # split categorical and continuous variables
    categorical_cols = []
    continuous_cols = []
    categorical_cols.extend(ordinal_cols)
    categorical_cols.extend(define_categorical_cols)
    continuous_cols.extend(define_continuous_cols)
    categorical_cols = list(set(categorical_cols))
    continuous_cols = list(set(continuous_cols))
    if str.lower(learning_type) == 'supervised':
        for col in dataset:
            if col != target_variable:
                if helper.check_categorical_col(dataset[col],
                                                categorical_threshold=categorical_threshold) and col not in reserved_cols:
                    categorical_cols.append(col)
                elif helper.check_numeric_col(dataset[col]) and col not in reserved_cols:
                    continuous_cols.append(col)
    else:
        for col in dataset:
            if helper.check_categorical_col(dataset[col],
                                            categorical_threshold=categorical_threshold) and col not in reserved_cols:
                categorical_cols.append(col)
            elif helper.check_numeric_col(dataset[col]) and col not in reserved_cols:
                continuous_cols.append(col)
    print('Columns identified as continuous are ' + ','.join(continuous_cols))
    print('Columns identified as categorical are ' + ','.join(categorical_cols))
    categorical_dataset = dataset[categorical_cols]
    continuous_dataset = dataset[continuous_cols]

    # encoding categorical features
    categorical_dataset = prepdata.impute_nulls(categorical_dataset)
    if str.lower(tranform_categorical) == 'label_encoding':
        encoded_labels, categorical_dataset = prepdata.get_label_encoded_df(categorical_dataset,
                                                                            categorical_threshold=categorical_threshold)
        col_labels.update(encoded_labels)
    elif str.lower(tranform_categorical) == 'one_hot_encoding':
        categorical_dataset = prepdata.get_ohe_df(categorical_dataset, ignore_cols=ohe_ignore_cols,
                                                  categorical_threshold=categorical_threshold)
        for col in ohe_ignore_cols:
            if helper.check_numeric_col(categorical_dataset[col]):
                pass
            else:
                label_dict, categorical_dataset = prepdata.label_encode(categorical_dataset, col)
                col_labels[col] = label_dict

    # impute nulls in continuous features using chosen method
    continuous_dataset = prepdata.impute_nulls(continuous_dataset, method=impute_null_method)

    # does log transform based on the chosen method if opted
    if log_transform is not None:
        continuous_dataset = prepdata.log_transform(dataset, method=log_transform,
                                                    categorical_threshold=categorical_threshold)

    # merge datasets
    cleansed_dataset = pd.concat([categorical_dataset, continuous_dataset], axis=1)
    if str.lower(learning_type) == 'supervised':
        target_df = pd.DataFrame(dataset[target_variable])
        if str.lower(target_type) == 'categorical':
            if not helper.check_numeric_col(dataset[target_variable]):
                # label encode if target variable is categorical
                label_dict, target_df = prepdata.label_encode(target_df, target_variable)
                col_labels[target_variable] = label_dict
        mapped_dataset = pd.concat([cleansed_dataset, target_df], axis=1)

        # remove outliers if opted
        if remove_outliers:
            mapped_dataset = prepdata.auto_remove_outliers(mapped_dataset, ignore_cols=[target_variable],
                                                           categorical_threshold=categorical_threshold)

        # does feature selection for supervised learning if opted
        if feature_selection:
            col_corr, correlated_features = prepdata.get_correlated_features(mapped_dataset, target_variable,
                                                                             target_type)
            final_dataset = mapped_dataset[correlated_features + [target_variable]]
        else:
            final_dataset = mapped_dataset
    elif str.lower(learning_type) == 'unsupervised':

        # remove outliers if opted
        if remove_outliers:
            cleansed_dataset = prepdata.auto_remove_outliers(cleansed_dataset,
                                                             categorical_threshold=categorical_threshold)
        final_dataset = cleansed_dataset
    return col_labels, final_dataset


def scale_transform(dataset,
                    method='standard'):
    """
    Usage: [arg1]:[dataframe], [method (default=standard)]:[Choose between standard, mimmax, robust, maxabs]
    Returns: numpy array [to be passed directly to ML model]
    |
    standard: Transorms data by removing mean
    mimmax: Fits values to a range around 0 to 1
    robust: Scaling data with outliers
    maxabs: Handling sparse data
    
    """
    if str.lower(method) == 'mimmax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    elif str.lower(method) == 'standard':
        scaler = sklearn.preprocessing.StandardScaler()
    elif str.lower(method) == 'robust':
        scaler = sklearn.preprocessing.RobustScaler()
    elif str.lower(method) == 'maxabs':
        scaler = sklearn.preprocessing.MaxAbsScaler()
    arr_data = scaler.fit_transform(dataset)
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '_')
    artifact_file = 'scaler_' + timestamp + '.pkl'
    dump(scaler, open(artifact_file, 'wb'))
    print(
        'Scaler artifact stored in ' + artifact_file + '. To reuse, execute scaler = load(open(<Scaler artifact file name>, \'rb\'))')
    return arr_data


def master_correlation(dataset,
                       define_continuous_cols=[],
                       define_nominal_cols=[],
                       define_ordinal_cols=[],
                       categorical_threshold=0.3,
                       impute_nulls=True,
                       only_target=True,
                       target_column=None,
                       target_type=None):
    """
    Usage:
    dataset=pandas DataFrame (required)
    categorical_threshold=0.3 (optional) [Threshold for determining categorical column based on the percentage of unique values]
    define_continuous_cols=[] (optional) [List - Predefine continuous variables]
    define_categorical_cols=[] (optional) [List - Predefine categorical variables]
    |
    Description: Auto-detects the type of data. Uses Pearson correlation between two continuous variables, CramersV correlation between two categorical variables, Kendalls Tau correlation between a categorical and a continuos variable
    |
    returns Correlation DataFrame
    """
    if only_target and target_column is None:
        raise exceptions.ParameterError('target_column is a required parameter if only_target is True')
    nominal_cols = []
    ordinal_cols = []
    continuous_cols = []
    nominal_cols.extend(define_nominal_cols)
    continuous_cols.extend(define_continuous_cols)
    ordinal_cols.extend(define_ordinal_cols)
    if target_type is not None:
        if target_type == 'continuous' and target_column not in continuous_cols:
            continuous_cols.append(target_column)
        elif target_type in ['nominal', 'categorical'] and target_column not in nominal_cols:
            nominal_cols.append(target_column)
        elif target_type == 'ordinal' and target_column not in ordinal_cols:
            ordinal_cols.append(target_column)
    dataset = prepdata.drop_single_valued_cols(dataset)
    for col in dataset:
        if col not in nominal_cols + ordinal_cols + continuous_cols:
            if helper.check_categorical_col(dataset[col], categorical_threshold=categorical_threshold):
                nominal_cols.append(col)
            elif helper.check_numeric_col(dataset[col]):
                continuous_cols.append(col)

    nominal_dataset = dataset[nominal_cols]
    ordinal_dataset = dataset[ordinal_cols]
    continuous_dataset = dataset[continuous_cols]
    print('Columns identified as ordinal are ' + ','.join(ordinal_cols))
    print('Columns identified as nominal are ' + ','.join(nominal_cols))
    print('Columns identified as continuous are ' + ','.join(continuous_cols))
    _, nominal_dataset = prepdata.get_label_encoded_df(nominal_dataset)

    data = pd.concat([nominal_dataset, continuous_dataset, ordinal_dataset], axis=1)
    if impute_nulls:
        data = prepdata.impute_nulls(data, method='central_tendency', define_continuous_cols=continuous_cols,
                                     define_nominal_cols=nominal_cols, define_ordinal_cols=ordinal_cols)

    column_list = data.columns

    column_combination = list(combinations(column_list, 2))

    corr_df = pd.DataFrame(columns=column_list, index=column_list)

    for col in column_list:
        corr_df.loc[col, col] = 1

    for comb in column_combination:
        if only_target and target_column not in comb:
            continue
        col1 = comb[0]
        col2 = comb[1]
        if col1 in continuous_cols and col2 in continuous_cols:
            corr_value1 = prepdata.pearson_corr(data[col1], data[col2])
            corr_value2 = prepdata.spearmans_corr(data[col2], data[col1])
            corr_value = max([corr_value1, corr_value2])
        elif col1 in continuous_cols and col2 in nominal_cols:
            corr_value = prepdata.nominal_scale_corr(data[col2], data[col1])
        elif col1 in continuous_cols and col2 in ordinal_cols:
            corr_value = prepdata.spearmans_corr(data[col2], data[col1])
        elif col1 in nominal_cols and col2 in nominal_cols:
            corr_value = prepdata.cramersv_corr(data[col1], data[col2])
        elif col1 in nominal_cols and col2 in continuous_cols:
            corr_value = prepdata.nominal_scale_corr(data[col1], data[col2])
        elif col1 in nominal_cols and col2 in ordinal_cols:
            corr_value = prepdata.cramersv_corr(data[col1], data[col2])
        elif col1 in ordinal_cols and col2 in ordinal_cols:
            corr_value = prepdata.kendalltau_corr(data[col1], data[col2])
        elif col1 in ordinal_cols and col2 in continuous_cols:
            corr_value = prepdata.spearmans_corr(data[col1], data[col2])
        elif col1 in ordinal_cols and col2 in nominal_cols:
            corr_value = prepdata.cramersv_corr(data[col1], data[col2])
        corr_df.loc[col1, col2] = corr_value
        corr_df.loc[col2, col1] = corr_value
    if only_target:
        corr_df = pd.DataFrame(corr_df[target_column])
    return corr_df


def feature_selection(dataset,
                      correlation_threshold=None,
                      select_top=None,
                      define_continuous_cols=[],
                      define_nominal_cols=[],
                      define_ordinal_cols=[],
                      categorical_threshold=0.3,
                      impute_nulls=True,
                      target_column=None,
                      target_type=None):
    corr_df = master_correlation(dataset, define_continuous_cols=define_continuous_cols,
                                 define_nominal_cols=define_nominal_cols, define_ordinal_cols=define_ordinal_cols,
                                 categorical_threshold=categorical_threshold, impute_nulls=impute_nulls,
                                 only_target=True, target_column=target_column, target_type=target_type)
    corr_dict = corr_df[target_column].to_dict()
    if correlation_threshold is None:
        correlation_threshold = 2 / np.sqrt(dataset.shape[0])
    selected_features = []
    for col in corr_dict.keys():
        if helper.get_absolute(corr_dict[col]) >= correlation_threshold:
            selected_features.append(col)
    if select_top is not None:
        selected_features = selected_features[:select_top]
    return selected_features


class Preprocessor:
    impute_null_method = 'central_tendency'
    tranform_categorical = 'label_encoding'
    categorical_threshold = 0.3
    remove_outliers = False
    log_transform = None
    drop_null_dominated = True
    dropna_threshold = 0.7
    derive_from_datetime = True
    ohe_ignore_cols = []
    ohe_drop_first = True
    feature_selection = True
    ordinal_dict = dict()
    artifact = dict()
    col_labels = dict()
    feature_selection_threshold = None
    feature_selection_top = None
    handle_outlier_ignore_cols = []
    ordinal_cols = []
    nominal_cols = []
    continuous_cols = []
    skewed_cols = []
    dataset_summary = dict()

    def __init__(self,
                 dataset,
                 learning_type,
                 target_variable=None,
                 target_type=None, ):
        self.dataset = dataset
        self.learning_type = learning_type
        self.target_variable = target_variable
        self.target_type = target_type

    def set_impute_null_method(self, impute_null_method):
        self.impute_null_method = impute_null_method

    def set_tranform_categorical(self, tranform_categorical, ohe_ignore_cols=[], ohe_drop_first=True):
        self.tranform_categorical = tranform_categorical
        self.ohe_ignore_cols = ohe_ignore_cols
        self.ohe_drop_first = ohe_drop_first

    def set_categorical_threshold(self, categorical_threshold):
        self.categorical_threshold = categorical_threshold

    def set_handle_outliers(self, handle_outliers, ignore_cols=[]):
        if str.lower(handle_outliers) == 'remove':
            self.remove_outliers = True
            self.handle_outlier_ignore_cols = ignore_cols

    def set_log_transform(self, log_transform):
        self.log_transform = log_transform

    def set_drop_null_dominated(self, drop_null_dominated, dropna_threshold=0.7):
        self.drop_null_dominated = drop_null_dominated
        self.dropna_threshold = dropna_threshold

    def derive_from_datetime(self, derive_from_datetime):
        self.derive_from_datetime = derive_from_datetime

    def set_feature_selection(self, correlation_check=True, threshold=None, select_top=None):
        self.feature_selection = correlation_check
        self.feature_selection_threshold = threshold
        self.feature_selection_top = select_top

    def set_continuous_columns(self, continuous_cols):
        self.continuous_cols = continuous_cols

    def set_nominal_columns(self, categorical_cols):
        self.nominal_cols = categorical_cols

    def set_ordinal_dict(self, ordinal_dict):
        self.ordinal_dict = ordinal_dict
        self.ordinal_cols.extend(list(ordinal_dict.keys()))
        self.col_labels.update(ordinal_dict)

    def get_preprocessor_artifact(self):
        return self.artifact

    def get_col_labels(self):
        return self.col_labels

    def get_processed_dataset(self):
        if str.lower(self.learning_type) not in ['supervised', 'unsupervised']:
            raise exceptions.ParameterError('learning_type should be supervised/unsupervised')
        if str.lower(self.tranform_categorical) not in ['label_encoding', 'one_hot_encoding']:
            raise exceptions.ParameterError('tranform_categorical should be label_encoding/one_hot_encoding')
        if str.lower(self.learning_type) == 'supervised' and self.target_variable is None:
            raise exceptions.ParameterError('target_variable is a required parameter for supervised learning')
        if str.lower(self.learning_type) == 'supervised' and self.target_type is None:
            raise exceptions.ParameterError(
                'target_type (continuous/categorical) is a required parameter for supervised learning')

        # resetting the index of the dataset
        self.dataset = self.dataset.reset_index(drop=True)

        # will be execute if derive_from_datetime is True
        if self.derive_from_datetime:
            self.dataset = prepdata.derive_from_datetime(self.dataset)

        # remove null dominated fields based on threshold if the flag is true
        if self.drop_null_dominated:
            self.dataset = prepdata.drop_null_fields(self.dataset, dropna_threshold=self.dropna_threshold)

        self.dataset = helper.bool_to_int(self.dataset)

        # drop all single valued columns
        self.dataset = prepdata.drop_single_valued_cols(self.dataset)

        # transform ordinal columns to integer values
        ordinal_labels, self.dataset = prepdata.get_ordinal_encoded_df(self.dataset,
                                                                       custom_ordinal_dict=self.ordinal_dict)
        self.col_labels.update(ordinal_labels)
        self.ordinal_cols.extend([col for col in ordinal_labels.keys() if col != self.target_variable])

        for col in self.dataset:
            if col != self.target_variable and col not in self.ordinal_cols + self.nominal_cols + self.continuous_cols:
                if helper.check_categorical_col(self.dataset[col], categorical_threshold=self.categorical_threshold):
                    self.nominal_cols.append(col)
                elif helper.check_numeric_col(self.dataset[col]):
                    self.continuous_cols.append(col)

        self.ordinal_cols = list(set(self.ordinal_cols))
        self.nominal_cols = list(set(self.nominal_cols))
        self.continuous_cols = list(set(self.continuous_cols))

        print('Columns identified as ordinal are ' + ','.join(self.ordinal_cols))
        print('Columns identified as nominal are ' + ','.join(self.nominal_cols))
        print('Columns identified as continuous are ' + ','.join(self.continuous_cols))

        summary_dict = prepdata.dataset_summary(
            self.dataset[self.continuous_cols + self.ordinal_cols + self.nominal_cols],
            define_continuous_cols=self.continuous_cols,
            define_nominal_cols=self.nominal_cols,
            define_ordinal_cols=self.ordinal_cols,
            categorical_threshold=self.categorical_threshold)
        self.dataset_summary.update(summary_dict)
        print(self.dataset_summary)

        if str.lower(self.tranform_categorical) == 'label_encoding':
            labels, self.dataset = prepdata.get_label_encoded_df(self.dataset,
                                                                 categorical_threshold=self.categorical_threshold,
                                                                 define_nominal_cols=self.nominal_cols,
                                                                 ignore_cols=self.ordinal_cols + self.continuous_cols)
            self.col_labels.update(labels)
        elif str.lower(self.tranform_categorical) == 'one_hot_encoding':
            self.dataset = prepdata.get_ohe_df(self.dataset,
                                               target_variable=self.target_variable,
                                               define_nominal_cols=self.nominal_cols,
                                               ignore_cols=self.ohe_ignore_cols + self.ordinal_cols + self.continuous_cols,
                                               categorical_threshold=self.categorical_threshold,
                                               drop_first=self.ohe_drop_first)
            if len(self.ohe_ignore_cols) > 0:
                labels, self.dataset = prepdata.get_label_encoded_df(self.dataset,
                                                                     categorical_threshold=self.categorical_threshold,
                                                                     define_nominal_cols=self.ohe_ignore_cols,
                                                                     ignore_cols=self.ordinal_cols + self.continuous_cols)
                self.col_labels.update(labels)

        self.dataset = prepdata.impute_nulls(self.dataset,
                                             method=self.impute_null_method,
                                             define_continuous_cols=self.continuous_cols,
                                             define_nominal_cols=self.nominal_cols,
                                             define_ordinal_cols=self.ordinal_cols,
                                             categorical_threshold=self.categorical_threshold)

        nominal_dataset = self.dataset[self.nominal_cols]
        ordinal_dataset = self.dataset[self.ordinal_cols]
        continuous_dataset = self.dataset[self.continuous_cols]
