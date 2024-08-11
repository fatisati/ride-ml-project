import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from tqdm import tqdm

# Initialize tqdm
tqdm.pandas()

# Simplified text preprocessing function
def preprocess_persian_text_simple(comment):
    return comment  # Use the raw comment without complex preprocessing

# Function to preprocess the `Created_at` column
def preprocess_datetime(df, column):
    df[column] = pd.to_datetime(df[column])
    df['Year'] = df[column].dt.year
    df['Month'] = df[column].dt.month
    df['Day'] = df[column].dt.day
    df['Hour'] = df[column].dt.hour
    # Drop the original datetime and unused columns
    df = df.drop(columns=[column, 'Minute', 'Second'], errors='ignore')  
    return df

# Function to handle categorical columns (encoding)
def encode_categorical(df, categorical_columns):
    df[categorical_columns] = df[categorical_columns].apply(lambda col: col.fillna('missing'))
    
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    
    return df

# Function to check and replace inf or large values
def check_and_replace_inf(df, columns):
    for col in columns:
        df[col].replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
        df[col].fillna(df[col].max(), inplace=True)  # Replace NaN (originally inf) with max value
        df[col] = np.clip(df[col], a_min=None, a_max=np.finfo(np.float64).max)  # Cap very large values
    return df

# Function to preprocess data
def preprocess_data(train_path, test_path, output_train_path, output_test_path):
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Preprocessing the 'Created_at' column...")
    # Preprocess the 'Created_at' column to extract date-time components
    train_df = preprocess_datetime(train_df, 'Created_at')
    test_df = preprocess_datetime(test_df, 'Created_at')

    print("Encoding categorical columns...")
    # Encode categorical columns (e.g., Time_Bucket, Duration_Category, Month, Day, Hour)
    categorical_columns = ['Time_Bucket', 'Duration_Category', 'Month', 'Day', 'Hour', 'Year']
    train_df = encode_categorical(train_df, categorical_columns)
    test_df = encode_categorical(test_df, categorical_columns)

    print("Preprocessing the 'Comment' column...")
    # No complex preprocessing, just a pass-through
    train_df['Processed_Comment'] = train_df['Comment'].progress_apply(preprocess_persian_text_simple)
    test_df['Processed_Comment'] = test_df['Comment'].progress_apply(preprocess_persian_text_simple)

    print("Vectorizing the processed comments using CountVectorizer...")
    # Use CountVectorizer for fast text encoding
    vectorizer = CountVectorizer(max_features=1000)  # Limit features to 1000 for speed
    X_train_counts = vectorizer.fit_transform(train_df['Processed_Comment']).toarray()
    X_test_counts = vectorizer.transform(test_df['Processed_Comment']).toarray()

    # Identify relevant columns for scaling
    columns_to_scale = ['Income', 'Income_Per_Minute', 'Income_Capped', 'Time']
    other_columns = [col for col in train_df.columns if col not in columns_to_scale + ['Comment', 'Processed_Comment', 'Label']]
    
    print("Checking and replacing inf values...")
    # Check and replace inf values
    train_df = check_and_replace_inf(train_df, columns_to_scale)
    test_df = check_and_replace_inf(test_df, columns_to_scale)

    print("Scaling relevant features...")
    # Scale only the relevant features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[columns_to_scale])
    X_test_scaled = scaler.transform(test_df[columns_to_scale])

    # Combine scaled features with non-scaled features and vectorized comments
    X_train = np.hstack([train_df[other_columns].values, X_train_scaled, X_train_counts])
    X_test = np.hstack([test_df[other_columns].values, X_test_scaled, X_test_counts])
    
    y_train = train_df['Label'].values
    y_test = test_df['Label'].values

    print("Saving the processed data...")
    # Save the processed data
    np.save(output_train_path + "_X.npy", X_train)
    np.save(output_train_path + "_y.npy", y_train)
    np.save(output_test_path + "_X.npy", X_test)
    np.save(output_test_path + "_y.npy", y_test)
    joblib.dump(scaler, output_train_path + "_scaler.pkl")
    joblib.dump(vectorizer, output_train_path + "_vectorizer.pkl")
    
    print(f"Data preprocessing completed and saved to {output_train_path} and {output_test_path}.")

if __name__ == "__main__":
    preprocess_data('data/task_train_processed.csv', 'data/task_test_processed.csv',
                    'data/processed_train', 'data/processed_test')
