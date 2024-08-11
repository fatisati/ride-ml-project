import pandas as pd
import numpy as np
from hazm import Normalizer, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def preprocess_persian_comment(comment):
    """Normalize and tokenize Persian text."""
    normalizer = Normalizer()
    normalized_text = normalizer.normalize(comment)  # Normalize the text
    tokens = word_tokenize(normalized_text)  # Tokenize the text into words
    return ' '.join(tokens)  # Join tokens back into a string

def preprocess_data(train_path, test_path, output_train_path, output_test_path):
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Preprocess the 'Comment' column using hazm
    train_df['Processed_Comment'] = train_df['Comment'].apply(preprocess_persian_comment)
    test_df['Processed_Comment'] = test_df['Comment'].apply(preprocess_persian_comment)

    # Use TF-IDF to vectorize the processed comments
    vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
    X_train_tfidf = vectorizer.fit_transform(train_df['Processed_Comment']).toarray()
    X_test_tfidf = vectorizer.transform(test_df['Processed_Comment']).toarray()

    # Combine the TF-IDF features with the original features (excluding 'Comment' and 'Processed_Comment')
    X_train = np.hstack([train_df.drop(columns=['Comment', 'Processed_Comment', 'Label']).values, X_train_tfidf])
    X_test = np.hstack([test_df.drop(columns=['Comment', 'Processed_Comment', 'Label']).values, X_test_tfidf])
    
    y_train = train_df['Label'].values
    y_test = test_df['Label'].values

    # Encode categorical features
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        combined_values = pd.concat([train_df[col], test_df[col]]).unique()
        le.fit(combined_values)
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
