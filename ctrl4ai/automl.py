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
from pickle import dump
from datetime import datetime

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
        print('learning_type should be supervised/unsupervised')
        raise exceptions.ParameterError
    if str.lower(tranform_categorical) not in ['label_encoding', 'one_hot_encoding']:
        print('learning_type should be label_encoding/one_hot_encoding')
        raise exceptions.ParameterError
    if str.lower(learning_type) == 'supervised' and target_variable is None:
        print('target_variable is a required parameter for supervised learning')
        raise exceptions.ParameterError
    if str.lower(learning_type) == 'supervised' and target_type is None:
        print('target_type (continuous/categorical) is a required parameter for supervised learning')
        raise exceptions.ParameterError

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
        continuous_dataset = prepdata.log_transform(method=log_transform, categorical_threshold=categorical_threshold)

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
                       categorical_threshold=0.3,
                       define_continuous_cols=[],
                       define_categorical_cols=[]):
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
    categorical_cols = []
    continuous_cols = []
    categorical_cols.extend(define_categorical_cols)
    continuous_cols.extend(define_continuous_cols)
    for col in dataset:
        if col not in categorical_cols + continuous_cols:
            if helper.check_categorical_col(dataset[col], categorical_threshold=categorical_threshold):
                categorical_cols.append(col)
            elif helper.check_numeric_col(dataset[col]):
                continuous_cols.append(col)

    categorical_dataset = dataset[categorical_cols]
    continuous_dataset = dataset[continuous_cols]

    _, categorical_dataset = prepdata.get_label_encoded_df(dataset[categorical_cols])

    data = pd.concat([categorical_dataset, continuous_dataset], axis=1)
    data = prepdata.drop_single_valued_cols(data)
    data = prepdata.impute_nulls(data, method='central_tendency')

    column_list = data.columns

    from itertools import combinations

    column_combination = list(combinations(column_list, 2))

    corr_df = pd.DataFrame(columns=column_list, index=column_list)

    for col in column_list:
        corr_df.loc[col, col] = 1

    for comb in column_combination:
        col1 = comb[0]
        col2 = comb[1]
        if col1 in continuous_cols and col2 in continuous_cols:
            corr_value = prepdata.pearson_corr(data[col1], data[col2])
        elif col1 in categorical_cols and col2 in categorical_cols:
            corr_value = prepdata.cramersv_corr(data[col1], data[col2])
        elif col1 in continuous_cols and col2 in categorical_cols:
            corr_value = prepdata.kendalltau_corr(data[col1], data[col2])
        elif col1 in categorical_cols and col2 in continuous_cols:
            corr_value = prepdata.kendalltau_corr(data[col1], data[col2])
        corr_df.loc[col1, col2] = corr_value
        corr_df.loc[col2, col1] = corr_value
    return corr_df


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
    feature_selection = True
    define_continuous_cols = []
    define_ordinal_cols = []
    define_nominal_cols = []
    ordinal_dict = dict()
    artifact = dict()
    col_labels = dict()
    feature_selection_threshold = None

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

    def set_tranform_categorical(self, tranform_categorical, ohe_ignore_cols=[]):
        self.tranform_categorical = tranform_categorical
        self.ohe_ignore_cols = ohe_ignore_cols

    def set_categorical_threshold(self, categorical_threshold):
        self.categorical_threshold = categorical_threshold

    def set_remove_outliers(self, remove_outliers):
        self.remove_outliers

    def set_log_transform(self, log_transform):
        self.log_transform = log_transform

    def set_drop_null_dominated(self, drop_null_dominated, dropna_threshold=0.7):
        self.drop_null_dominated = drop_null_dominated
        self.dropna_threshold = dropna_threshold

    def derive_from_datetime(self, derive_from_datetime):
        self.derive_from_datetime = derive_from_datetime

    def set_feature_selection(self, feature_selection, threshold=None):
        self.feature_selection = feature_selection
        self.feature_selection_threshold = threshold

    def set_continuous_columns(self, continuous_cols):
        self.define_continuous_cols = continuous_cols

    def set_nominal_columns(self, categorical_cols):
        self.define_nominal_cols = categorical_cols

    def set_ordinal_dict(self, ordinal_dict):
        self.ordinal_dict = ordinal_dict
        self.define_ordinal_cols.extend(list(ordinal_dict.keys()))
        self.col_labels.update(ordinal_dict)

    def get_preprocessor_artifact(self, path):
        return self.artifact

    def get_col_labels(self):
        return self.col_labels

    def get_processed_dataset(self):
        if str.lower(self.learning_type) not in ['supervised', 'unsupervised']:
            print('learning_type should be supervised/unsupervised')
            raise exceptions.ParameterError
        if str.lower(self.tranform_categorical) not in ['label_encoding', 'one_hot_encoding']:
            print('tranform_categorical should be label_encoding/one_hot_encoding')
            raise exceptions.ParameterError
        if str.lower(self.learning_type) == 'supervised' and self.target_variable is None:
            print('target_variable is a required parameter for supervised learning')
            raise exceptions.ParameterError
        if str.lower(self.learning_type) == 'supervised' and self.target_type is None:
            print('target_type (continuous/categorical) is a required parameter for supervised learning')
            raise exceptions.ParameterError

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
        self.ordinal_labels, self.dataset = prepdata.get_ordinal_encoded_df(self.dataset)

        self.col_labels.update(self.ordinal_labels)
        self.ordinal_cols = [col for col in self.ordinal_labels.keys() if col != self.target_variable]
        print('Columns identified as ordinal are ' + ','.join(self.ordinal_cols))
        reserved_cols = []
        reserved_cols.extend(define_continuous_cols)
        reserved_cols.extend(define_categorical_cols)
        reserved_cols.extend(ordinal_cols)

        # split nominal and continuous variables
        continuous_cols = []
        categorical_cols.extend(ordinal_cols)
        categorical_cols.extend(define_categorical_cols)
        continuous_cols.extend(define_continuous_cols)
        categorical_cols = list(set(categorical_cols))
        continuous_cols = list(set(continuous_cols))