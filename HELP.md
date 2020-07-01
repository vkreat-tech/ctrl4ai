# ctrl4ai.automl
    
##    preprocess(dataset, learning_type, target_variable=None, target_type=None, impute_null_method='central_tendency', tranform_categorical='label_encoding', categorical_threshold=0.3, remove_outliers=False, log_transform=None, drop_null_dominated=True, dropna_threshold=0.7, derive_from_datetime=True, ohe_ignore_cols=[], feature_selection=True, define_continuous_cols=[], define_categorical_cols=[])
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

##    master_correlation(dataset, categorical_threshold=0.3, define_continuous_cols=[], define_categorical_cols=[])
        Usage:
        dataset=pandas DataFrame (required)
        categorical_threshold=0.3 (optional) [Threshold for determining categorical column based on the percentage of unique values]
        define_continuous_cols=[] (optional) [List - Predefine continuous variables]
        define_categorical_cols=[] (optional) [List - Predefine categorical variables]
        |
        Description: Auto-detects the type of data. Uses Pearson correlation between two continuous variables, CramersV correlation between two categorical variables, Kendalls Tau correlation between a categorical and a continuos variable
        |
        returns Correlation DataFrame
    
##    scale_transform(dataset, method='standard')
        Usage: [arg1]:[dataframe], [method (default=standard)]:[Choose between standard, mimmax, robust, maxabs]
        Returns: numpy array [to be passed directly to ML model]
        |
        standard: Transorms data by removing mean
        mimmax: Fits values to a range around 0 to 1
        robust: Scaling data with outliers
        maxabs: Handling sparse data


# ctrl4ai.preprocessing

##    auto_remove_outliers(dataset, ignore_cols=[], categorical_threshold=0.3)
        Usage: [arg1]:[pandas dataframe],[ignore_cols]:[list of columns to be ignored],[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
        Description: Checks if the column is continuous and removes outliers
        Returns: DataFrame with outliers removed
    
##    cramersv_corr(x, y)
        Usage: [arg1]:[categorical series],[arg2]:[categorical series]
        Description: Cramer's V Correlation is a measure of association between two categorical variables
        Returns: A value between 0 and +1
    
##    derive_from_datetime(dataset)
        Usage: [arg1]:[pandas dataframe]
        Prerequisite: Type for datetime columns to be defined correctly
        Description: Derives the hour, weekday, year and month from a datetime column
        Returns: Dataframe [with new columns derived from datetime columns]
    
##    drop_non_numeric(dataset)
        Usage: [arg1]:[pandas dataframe]
        Description: Drop columns that are not numeric
        Returns: Dataframe [only numeric features]
    
##    drop_null_fields(dataset, dropna_threshold=0.7)
        Usage: [arg1]:[pandas dataframe],[dropna_threshold(default=0.7)]:[What percentage of nulls should account for the column top be removed]
        Description: Drop columns that has more null values
        Returns: Dataframe [with null dominated columns removed]
    
##    drop_single_valued_cols(dataset)
        Usage: [arg1]:[pandas dataframe]
        Description: Drop columns that has only one value in it
        Returns: Dataframe [without single valued columns]
    
##    get_correlated_features(dataset, target_col, target_type, categorical_threshold=0.3)
        Usage: [arg1]:[pandas dataframe],[arg2]:[target/dependent variable],[arg3]:['continuous'/'categorical'],,[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
        Description: Only for supervised learning to select independent variables that has some correlation with target/dependent variable (Uses Pearson correlation between two continuous variables, CramersV correlation between two categorical variables, Kendalls Tau correlation between a categorical and a continuos variable)
        Returns: Dictionary of correlation coefficients, List of columns that have considerable correlation
    
##    get_distance(dataset, start_latitude, start_longitude, end_latitude, end_longitude)
        Usage: [arg1]:[Pandas DataFrame],[arg2]:[column-start_latitude],[arg3]:[column-start_longitude],[arg4]:[column-end_latitude],[arg5]:[column-end_longitude]
        Returns: DataFrame with additional column [Distance in kilometers]
    
##    get_label_encoded_df(dataset, categorical_threshold=0.3)
        Usage: [arg1]:[pandas dataframe],[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
        Description: Auto identifies categorical features in the dataframe and does label encoding
        Returns: Dataframe [with separate column for each categorical values]
    
##    get_ohe_df(dataset, target_variable=None, ignore_cols=[], categorical_threshold=0.3)
        Usage: [arg1]:[pandas dataframe],[target_variable(default=None)]:[Dependent variable for Regression/Classification],[ignore_cols]:[categorical columns where one hot encoding need not be done],[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
        Description: Auto identifies categorical features in the dataframe and does one hot encoding
        Note: Consumes more system mermory if the size of the dataset is huge
        Returns: Dataframe [with separate column for each categorical values]
    
##    get_timediff(dataset, start_time, end_time)
        Usage: [arg1]:[Pandas DataFrame],[arg2]:[column-start_time],[arg3]:[column-end_time]
        Returns: DataFrame with additional column [Duration in seconds]
    
 ##   impute_nulls(dataset, method='central_tendency')
        Usage: [arg1]:[pandas dataframe],[method(default=central_tendency)]:[Choose either central_tendency or KNN]
        Description: Auto identifies the type of distribution in the column and imputes null values
        Note: KNN consumes more system mermory if the size of the dataset is huge
        Returns: Dataframe [with separate column for each categorical values]
    
##    kendalltau_corr(x, y)
        Usage: [arg1]:[continuous series],[arg2]:[categorical series]
        Description: Kendall Tau Correlation is a measure of association between a continuous variable and a categorical variable
        Returns: A value between -1 and +1
    
##    label_encode(dataset, col)
        Usage: [arg1]:[pandas dataframe],[arg1]:[column to be encoded]
        Description: Labelling categorical features with numbers from 0 to n categories
        Returns: Label Dict , Dataframe
    
##    log_transform(dataset, method='yeojohnson', categorical_threshold=0.3)
        Usage: [arg1]:[pandas dataframe],[method]=['yeojohnson'/'added_constant']
        Description: Checks if the a continuous column is skewed and does log transformation
        Returns: Dataframe [with all skewed columns normalized using appropriate approach]
    
##    pearson_corr(x, y)
        Usage: [arg1]:[continuous series],[arg2]:[continuous series]
        Description: Pearson Correlation is a measure of association between two continuous variables
        Returns: A value between -1 and +1
    
##    remove_outlier_df(dataset, cols)
        Usage: [arg1]:[pandas dataframe],[arg2]:[list of columns to check and remove outliers]
        Description: The column needs to be continuous
        Returns: DataFrame with outliers removed for the specific columns


#   ctrl4ai.helper

##    added_constant_log(dataset, col)
        Usage: [arg1]:[dataset], [arg2]:[column in which log transform should be done]
        Description: Log transforms the specified column
        Returns: DataFrame
    
 ##   check_categorical_col(col_series, categorical_threshold=0.3)
        Usage: [arg1]:[Pandas Series / Single selected column of a dataframe],[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
        Description: Breaks the values to chunks and checks if the proportion of unique values is less than the threshold
        Returns: Boolean [True/False]
    
##    check_numeric_col(col_series)
        Usage: [arg1]:[Pandas Series / Single selected column of a dataframe]
        Description: Checks if all the values in the series are numerical
        Returns: Boolean [True/False]
    
##    distance_calculator(start_latitude, start_longitude, end_latitude, end_longitude)
        Usage: [arg1]:[numeric-start_latitude],[arg2]:[numeric-start_longitude],[arg3]:[numeric-end_latitude],[arg4]:[numeric-end_longitude]
        Returns: Numeric [Distance in kilometers]
    
##    isNaN(num)
        Usage: [arg1]:[numeric value]
        Description: Checks if the value is null (numpy.NaN)
        Returns: Boolean [True/False]
    
##    one_hot_encoding(dataset, categorical_cols_list)
        Usage: [arg1]:[pandas dataframe],[arg2]:[list of columns to be encoded]
        Description: Transformation for categorical features by getting dummies
        Returns: Dataframe [with separate column for each categorical values]
    
##    single_valued_col(col_series)
        Usage: [arg1]:[Pandas Series / Single selected column of a dataframe]
        Description: Checks if the column has only one value
        Returns: Boolean [True/False]
    
##    test_numeric(test_string)
        Usage: [arg1]:[String/Number]
        Description: Checks if the value is numeric
        Returns: Boolean [True/False]
    
##    yeojohnsonlog(x)
        Usage: [arg1]:[real/float value]
        Description: Log transforms the specified column based on Yeo Joshson Power Transform
        Returns: Log value (numeric)