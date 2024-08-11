import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def preprocess_data(config):
    train_df = pd.read_csv(config['data']['train_path'])
    test_df = pd.read_csv(config['data']['test_path'])

    # Separate features and target
    X_train = train_df.drop(columns=['Label'])
    y_train = train_df['Label']
    X_test = test_df.drop(columns=['Label'])
    y_test = test_df['Label']

    # Encode categorical features
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test = preprocess_data(config)
