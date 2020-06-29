# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:00:36 2020

@author: Shaji,Charu,Selva
"""

from . import preprocessing
from . import helper
from . import exceptions


import pandas as pd
pd.set_option('mode.chained_assignment', None)

def preprocess(dataset,
               learning_type,
               target_variable=None,
               target_type=None,
               impute_null_method='central_tendency',
               tranform_categorical='label_encoding',
               categorical_threshold=0.3,
               remove_outliers=True,
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
    categorical_threshold=0.3 (optional) [Threshold for determing categorical column based on the percentage of unique values]
    remove_outliers=True (optional) [Choose between True and False]
    drop_null_dominated=True (optional) [Choose between True and False - Optionally change threshold in dropna_threshold if True]
    dropna_threshold=0.7 (optional) [Proportion check for dropping dull dominated column]
    derive_from_datetime=True (optional) [derive hour, year, month, weekday etc from datetime column - make sure that the dtype is datetime for the column]
    ohe_ignore_cols=[] (optional) [List - if tranform_categorical=one_hot_encoding, ignore columns not to be one hot encoded]
    feature_selection=True (optional) [Choose between True and False - Uses Pearson correlation between two continuous variables, CramersV correlation between two categorical variables, Kendalls Tau correlation between a categorical and a continuos variable]
    define_continuous_cols=[] (optional) [List - Predefine continuous variables]
    define_categorical_cols=[] (optional) [List - Predefine categorical variables]
    
    .
    .
    returns [Dict - Label Encoded Columns and Values], [DataFrame - Processed Dataset]
    """
    
    col_labels=dict()
    if str.lower(learning_type) not in ['supervised','unsupervised']:
        print('learning_type should be supervised/unsupervised')
        raise exceptions.ParameterError
    if str.lower(tranform_categorical) not in ['label_encoding','one_hot_encoding']:
        print('learning_type should be label_encoding/one_hot_encoding')
        raise exceptions.ParameterError
    if str.lower(learning_type)=='supervised' and target_variable==None:
        print('target_variable is a required parameter for supervised learning')
        raise exceptions.ParameterError
    if str.lower(learning_type)=='supervised' and target_type==None:
        print('target_type (continuous/categorical) is a required parameter for supervised learning')
        raise exceptions.ParameterError
    dataset=dataset.reset_index(drop=True)
    if derive_from_datetime:
        dataset=preprocessing.derive_from_datetime(dataset)
    if drop_null_dominated:
        dataset=preprocessing.drop_null_fields(dataset,dropna_threshold=dropna_threshold)
    dataset=preprocessing.drop_single_valued_cols(dataset)
    categorical_cols=[]
    continuous_cols=[]
    if str.lower(learning_type)=='supervised':
        for col in dataset:
            if col!=target_variable:
                if helper.check_categorical_col(dataset[col],categorical_threshold=categorical_threshold) and col not in define_continuous_cols:
                    categorical_cols.append(col)
                elif helper.check_numeric_col(dataset[col]) and col not in define_categorical_cols:
                    continuous_cols.append(col)
    else:
        for col in dataset:
            if helper.check_categorical_col(dataset[col],categorical_threshold=categorical_threshold) and col not in define_continuous_cols:
                categorical_cols.append(col)
            elif helper.check_numeric_col(dataset[col]) and col not in define_categorical_cols:
                continuous_cols.append(col)
    for col in define_categorical_cols:
        if col not in categorical_cols:
            categorical_cols.append(col)
    for col in define_continuous_cols:
        if col not in continuous_cols:
            continuous_cols.append(col)
    print('Columns identified as continuous are '+','.join(continuous_cols))        
    print('Columns identified as categorical are '+','.join(categorical_cols))   
    categorical_dataset=dataset[categorical_cols]
    continuous_dataset=dataset[continuous_cols]
    categorical_dataset=preprocessing.impute_nulls(categorical_dataset)
    if str.lower(tranform_categorical)=='label_encoding':
        col_labels,categorical_dataset=preprocessing.get_label_encoded_df(categorical_dataset,categorical_threshold=categorical_threshold)
    elif str.lower(tranform_categorical)=='one_hot_encoding':
        categorical_dataset=preprocessing.get_ohe_df(categorical_dataset,ignore_cols=ohe_ignore_cols,categorical_threshold=categorical_threshold)
        for col in ohe_ignore_cols:
            if helper.check_numeric_col(categorical_dataset[col]):
                pass
            else:
                label_dict,categorical_dataset=preprocessing.label_encode(categorical_dataset,col)
                col_labels[col]=label_dict
    continuous_dataset=preprocessing.impute_nulls(continuous_dataset,method=impute_null_method)
    cleansed_dataset=pd.concat([categorical_dataset,continuous_dataset],axis=1)
    if str.lower(learning_type)=='supervised':
        target_df=pd.DataFrame(dataset[target_variable])
        if str.lower(target_type)=='categorical':
            label_dict,target_df=preprocessing.label_encode(target_df,target_variable)
            col_labels[target_variable]=label_dict
        mapped_dataset=pd.concat([cleansed_dataset,target_df],axis=1)
        if remove_outliers:
            mapped_dataset=preprocessing.auto_remove_outliers(mapped_dataset,ignore_cols=[target_variable],categorical_threshold=categorical_threshold)
        if feature_selection:
            col_corr,correlated_features=preprocessing.get_correlated_features(mapped_dataset,target_variable,target_type)
            final_dataset=mapped_dataset[correlated_features+[target_variable]]
    elif str.lower(learning_type)=='unsupervised':
        if remove_outliers:
            cleansed_dataset=preprocessing.auto_remove_outliers(cleansed_dataset,categorical_threshold=categorical_threshold)
        final_dataset=cleansed_dataset
    return col_labels,final_dataset