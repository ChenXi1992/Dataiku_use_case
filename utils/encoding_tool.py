# Import necessary library
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def create_education_ordinal_embedding(education_level, education_hierarchy):
    """
    Creates an ordinal embedding for an education level based on a predefined hierarchy.

    Parameters:
    education_level (str): The individual's education level.
    education_hierarchy (list): A list representing the hierarchy of education levels in increasing order.

    Returns:
    list: A binary list encoding the education level as 1s for levels achieved and 0s otherwise.
    """

    try:
        # Find the index of the given education level in the hierarchy
        level_index = education_hierarchy.index(education_level)
    except ValueError:
        # If the education level is not found, return a zero-vector of the same length
        return [0] * len(education_hierarchy)
    
    # Create a binary embedding: 1 for levels achieved, 0 for levels not achieved
    embedding = [1 if i <= level_index else 0 for i in range(len(education_hierarchy))]
    
    return embedding


def engineer_household_features(df):
    """
    Engineers household-related features based on the 'detailed_household_and_family_stat' column.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the household and family status feature.

    Returns:
    pd.DataFrame: Transformed DataFrame with new household-related features.
    """

    # Create a copy of the DataFrame to avoid modifying the original data
    result = df.copy()
    
    # Extract primary relationship type
    result['rel_type'] = result['detailed_household_and_family_stat'].apply(
        lambda x: ('Child' if 'Child' in x else
                   'Grandchild' if 'Grandchild' in x else
                   'Spouse' if 'Spouse' in x else
                   'Householder' if 'Householder' in x else
                   'Other_Rel' if 'Other Rel' in x else
                   'Group_Quarters' if 'group quarters' in x else
                   'Secondary' if 'Secondary' in x else 'Unknown')
    )
    
    # Create binary feature for individuals under 18
    result['is_under_18'] = result['detailed_household_and_family_stat'].str.contains('<18').astype(int)
    
    # Create binary feature indicating whether the individual is in a subfamily
    result['in_subfamily'] = result['detailed_household_and_family_stat'].str.contains('not in subfamily').astype(int)
    
    # Create binary feature for whether the individual is the reference person in a household
    result['is_reference_person'] = result['detailed_household_and_family_stat'].str.contains('RP of').astype(int)
    
    # Marital status indicators
    result['ever_married'] = result['detailed_household_and_family_stat'].str.contains('ever marr').astype(int)
    result['never_married'] = result['detailed_household_and_family_stat'].str.contains('never marr').astype(int)
    
    # One-hot encode the primary relationship type
    rel_type_dummies = pd.get_dummies(result['rel_type'], prefix='rel')
    
    # Concatenate the new one-hot encoded columns to the DataFrame
    result = pd.concat([result, rel_type_dummies], axis=1)
    
    # Drop the original categorical feature
    result.drop(columns=['rel_type'], inplace=True)
    
    return result


        
def one_hot_encode_features(df_train, df_test, one_hot_enc_feat):
    """
    One-hot encodes the specified features in both training and test datasets.
    
    Parameters:
    df_train : Training dataset
    df_test : Test dataset
    one_hot_enc_feat : List of feature names to be one-hot encoded
        
    Returns:
    df_train : Training dataset with one-hot encoded features
    df_test : Test dataset with one-hot encoded features
    """
    for feature in one_hot_enc_feat:
        # Define encoder
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Fit on training data
        feature_array_train = df_train[feature].values.reshape(-1, 1)
        one_hot_encoder.fit(feature_array_train)  # Fit only on training data
        
        # Transform both training and test data
        encoded_train = one_hot_encoder.transform(feature_array_train)
        
        # Transform test data using the same encoder
        feature_array_test = df_test[feature].values.reshape(-1, 1)
        encoded_test = one_hot_encoder.transform(feature_array_test)
        
        # Convert to DataFrame
        feature_names = [f"{feature}_{val}" for val in one_hot_encoder.categories_[0]]
        encoded_train_df = pd.DataFrame(encoded_train, columns=feature_names, index=df_train.index)
        encoded_test_df = pd.DataFrame(encoded_test, columns=feature_names, index=df_test.index)
        
        # Drop original feature and concatenate encoded columns
        df_train = pd.concat([df_train.drop(feature, axis=1), encoded_train_df], axis=1)
        df_test = pd.concat([df_test.drop(feature, axis=1), encoded_test_df], axis=1)
        
    return df_train, df_test


def encode_ordinal_feature(df_train, df_test, column_name, hierarchy):
    """
    Encodes an ordinal categorical feature into multiple columns based on a given hierarchy.
    
    Parameters:
    df_train (pd.DataFrame): Training dataset.
    df_test (pd.DataFrame): Test dataset.
    column_name (str): Name of the column to encode.
    hierarchy (list): Ordered list of category levels.
    encoding_tool (object): Tool containing the encoding function.
    
    Returns:
    pd.DataFrame, pd.DataFrame: Updated train and test DataFrames with encoded features.
    """
    # Apply ordinal encoding function
    encoded_train = df_train[column_name].apply(lambda x: create_education_ordinal_embedding(x, hierarchy))
    encoded_test = df_test[column_name].apply(lambda x: create_education_ordinal_embedding(x, hierarchy))
    
    # Generate column names based on hierarchy
    col_names = [f"{column_name}_{level}" for level in hierarchy]
    
    # Convert embeddings to DataFrames
    encoded_df_train = pd.DataFrame(encoded_train.tolist(), columns=col_names, index=df_train.index)
    encoded_df_test = pd.DataFrame(encoded_test.tolist(), columns=col_names, index=df_test.index)
    
    # Drop original column and append encoded features
    df_train = df_train.drop(column_name, axis=1)
    df_test = df_test.drop(column_name, axis=1)
    df_train = pd.concat([df_train, encoded_df_train], axis=1)
    df_test = pd.concat([df_test, encoded_df_test], axis=1)
    
    return df_train, df_test
