import sys
import os
import numpy as np
import torch
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from imblearn.over_sampling import SMOTE

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import create_deep_nn_model  # Ensure this is correctly imported

def load_preprocessed_data(train_path, test_path):
    X_train = np.load(train_path + "_X.npy", allow_pickle=True)
    y_train = np.load(train_path + "_y.npy", allow_pickle=True)
    X_test = np.load(test_path + "_X.npy", allow_pickle=True)
    y_test = np.load(test_path + "_y.npy", allow_pickle=True)
    return X_train, y_train, X_test, y_test

def resample_data(X_train, y_train):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def evaluate_model_on_train_and_test(model, X_train, y_train, X_test, y_test):
    # Evaluate on training data
    y_train_pred = model.predict(X_train)
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1_score': f1_score(y_train, y_train_pred)
    }
    
    # Evaluate on test data
    y_test_pred = model.predict(X_test)
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred)
    }
    
    return train_metrics, test_metrics

def evaluate_deep_nn_on_train_and_test(model, X_train, y_train, X_test, y_test, device):
    model.eval()
    
    # Evaluate on training data
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_train_pred_prob = model(X_train_tensor).cpu().numpy()
    y_train_pred = (y_train_pred_prob > 0.5).astype(int)
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1_score': f1_score(y_train, y_train_pred)
    }
    
    # Evaluate on test data
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_test_pred_prob = model(X_test_tensor).cpu().numpy()
    y_test_pred = (y_test_pred_prob > 0.5).astype(int)
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred)
    }
    
    return train_metrics, test_metrics

def generate_metrics_dataframe(metrics_dict):
    metrics_df = pd.DataFrame(metrics_dict)
    return metrics_df

def evaluate_models():
    # Load preprocessed data
    X_train, y_train, X_test, y_test = load_preprocessed_data('data/processed_train', 'data/processed_test')

    # Resample data using SMOTE
    X_train_resampled, y_train_resampled = resample_data(X_train, y_train)

    # Load models
    log_reg_model = joblib.load('log_reg_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    
    # Create a new instance of the model and load the state dict
    input_dim = X_train_resampled.shape[1]  # Make sure input_dim matches the number of features
    deep_nn_model = create_deep_nn_model(input_dim=input_dim, num_classes=1, dropout=0.5)
    deep_nn_model.load_state_dict(torch.load('deep_nn_model.pth'))
    
    # Evaluate Logistic Regression
    log_reg_train_metrics, log_reg_test_metrics = evaluate_model_on_train_and_test(log_reg_model, X_train_resampled, y_train_resampled, X_test, y_test)

    # Evaluate Random Forest
    rf_train_metrics, rf_test_metrics = evaluate_model_on_train_and_test(rf_model, X_train_resampled, y_train_resampled, X_test, y_test)

    # Evaluate Deep Neural Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deep_nn_model.to(device)
    nn_train_metrics, nn_test_metrics = evaluate_deep_nn_on_train_and_test(deep_nn_model, X_train_resampled, y_train_resampled, X_test, y_test, device)

    # Generate metrics DataFrame
    metrics_dict = {
        'Model': ['Logistic Regression', 'Random Forest', 'Deep Neural Network'],
        'Train Accuracy': [log_reg_train_metrics['accuracy'], rf_train_metrics['accuracy'], nn_train_metrics['accuracy']],
        'Train Precision': [log_reg_train_metrics['precision'], rf_train_metrics['precision'], nn_train_metrics['precision']],
        'Train Recall': [log_reg_train_metrics['recall'], rf_train_metrics['recall'], nn_train_metrics['recall']],
        'Train F1 Score': [log_reg_train_metrics['f1_score'], rf_train_metrics['f1_score'], nn_train_metrics['f1_score']],
        'Test Accuracy': [log_reg_test_metrics['accuracy'], rf_test_metrics['accuracy'], nn_test_metrics['accuracy']],
        'Test Precision': [log_reg_test_metrics['precision'], rf_test_metrics['precision'], nn_test_metrics['precision']],
        'Test Recall': [log_reg_test_metrics['recall'], rf_test_metrics['recall'], nn_test_metrics['recall']],
        'Test F1 Score': [log_reg_test_metrics['f1_score'], rf_test_metrics['f1_score'], nn_test_metrics['f1_score']]
    }
    
    metrics_df = generate_metrics_dataframe(metrics_dict)
    return metrics_df
