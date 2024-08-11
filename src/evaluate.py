import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from src.models import create_logistic_regression_model, create_random_forest_model, create_deep_nn_model
import torch
import joblib
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(config):
    test_df = pd.read_csv(config['data']['test_path'])
    test_df = test_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    X_test = test_df.drop(columns=['Label'])
    y_test = test_df['Label']

    categorical_cols = X_test.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_test[col] = le.fit_transform(X_test[col])
        
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    
    return X_test, y_test

def evaluate_model(model, X_test, y_test, model_name):
    if model_name == 'deep_nn':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_deep_nn_model(X_test.shape[1], 1)
        model.load_state_dict(torch.load('deep_nn_model.pth'))
        model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_prob = model(X_test_tensor).squeeze().cpu().numpy()
            y_pred = (y_pred_prob > 0.5).astype(int)
    else:
        model = joblib.load(f'{model_name}_model.pkl')
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

    print(f"{model_name} Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_prob):.2f}")

if __name__ == "__main__":
    config = load_config()
    X_test, y_test = load_data(config)

    evaluate_model(None, X_test, y_test, 'logistic_regression')
    evaluate_model(None, X_test, y_test, 'random_forest')
    evaluate_model(None, X_test, y_test, 'deep_nn')
