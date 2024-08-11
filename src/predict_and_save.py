import numpy as np
import torch
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Add the parent directory of 'src' to the Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import create_deep_nn_model

def load_preprocessed_data(test_path):
    X_test = np.load(test_path + "_X.npy", allow_pickle=True)
    test_df = pd.read_csv(test_path + ".csv")  # Load the original test dataframe to preserve other columns
    return X_test, test_df

def load_models():
    log_reg_model = joblib.load('log_reg_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    
    input_dim = X_test.shape[1]
    deep_nn_model = create_deep_nn_model(input_dim=input_dim, num_classes=1, dropout=0.5)
    deep_nn_model.load_state_dict(torch.load('deep_nn_model.pth'))
    deep_nn_model.eval()
    
    xgb_model = joblib.load('xgb_model.pkl')
    
    return log_reg_model, rf_model, deep_nn_model, xgb_model

def make_predictions(log_reg_model, rf_model, deep_nn_model, xgb_model, X_test):
    # Logistic Regression
    log_reg_preds = log_reg_model.predict(X_test)
    
    # Random Forest
    rf_preds = rf_model.predict(X_test)
    
    # XGBoost
    xgb_preds = xgb_model.predict(X_test)
    
    # Deep Neural Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deep_nn_model.to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        deep_nn_preds_prob = deep_nn_model(X_test_tensor).cpu().numpy()
    deep_nn_preds = (deep_nn_preds_prob > 0.5).astype(int)
    
    return log_reg_preds, rf_preds, xgb_preds, deep_nn_preds

def save_predictions(test_df, log_reg_preds, rf_preds, xgb_preds, deep_nn_preds, output_path):
    # Add predictions to the test DataFrame
    test_df['Logistic_Regression_Pred'] = log_reg_preds
    test_df['Random_Forest_Pred'] = rf_preds
    test_df['XGBoost_Pred'] = xgb_preds
    test_df['Deep_NN_Pred'] = deep_nn_preds
    
    # Save the DataFrame to a CSV file
    test_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    # Load the preprocessed test data and the original test dataframe
    X_test, test_df = load_preprocessed_data('data/processed_test')

    # Load trained models
    log_reg_model, rf_model, deep_nn_model, xgb_model = load_models()

    # Make predictions
    log_reg_preds, rf_preds, xgb_preds, deep_nn_preds = make_predictions(log_reg_model, rf_model, deep_nn_model, xgb_model, X_test)

    # Save predictions to CSV
    save_predictions(test_df, log_reg_preds, rf_preds, xgb_preds, deep_nn_preds, 'data/test_predictions.csv')
