import pandas as pd
import numpy as np

def extract_day_feature(df, datetime_column='Time'):
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
    df['Day_of_Week'] = df[datetime_column].dt.day_name()
    return df

def add_custom_features(df):
    df['Income_Log'] = np.log1p(df['Income'])
    
    # Convert Time to total minutes (assuming the 'Time' column is a datetime object)
    df['Time_in_Minutes'] = df['Time'].dt.hour * 60 + df['Time'].dt.minute
    
    # Now perform the division
    df['Income_Per_Minute'] = df['Income'] / df['Time_in_Minutes']
    
    return df

def preprocess_and_engineer_features(df):
    df = extract_day_feature(df)
    df = add_custom_features(df)
    return df


if __name__ == "__main__":
    # Load the data
    train_df = pd.read_csv('data/task_train.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    test_df = pd.read_csv('data/task_test.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

    # Apply feature engineering to both datasets
    train_df = preprocess_and_engineer_features(train_df)
    test_df = preprocess_and_engineer_features(test_df)

    # Now you can proceed with further processing, model training, etc.
