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

# Function to preprocess data
def preprocess_data(train_path, test_path, output_train_path, output_test_path):
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Preprocessing the 'Comment' column...")
    # No complex preprocessing, just a pass-through
    train_df['Processed_Comment'] = train_df['Comment'].progress_apply(preprocess_persian_text_simple)
    test_df['Processed_Comment'] = test_df['Comment'].progress_apply(preprocess_persian_text_simple)

    print("Vectorizing the processed comments using CountVectorizer...")
    # Use CountVectorizer for fast text encoding
    vectorizer = CountVectorizer(max_features=1000)  # Limit features to 1000 for speed
    X_train_counts = vectorizer.fit_transform(train_df['Processed_Comment']).toarray()
    X_test_counts = vectorizer.transform(test_df['Processed_Comment']).toarray()

    print("Combining CountVectorizer features with original dataset features...")
    # Exclude non-numeric columns and combine the CountVectorizer features with numeric ones
    non_numeric_cols = ['Comment', 'Processed_Comment', 'Label']
    numeric_cols = train_df.drop(columns=non_numeric_cols).select_dtypes(include=[np.number]).columns
    X_train = np.hstack([train_df[numeric_cols].values, X_train_counts])
    X_test = np.hstack([test_df[numeric_cols].values, X_test_counts])
    
    y_train = train_df['Label'].values
    y_test = test_df['Label'].values

    print("Normalizing features...")
    # Normalize only the numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
