import numpy as np
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

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

def train_logistic_regression(X_train, y_train):
    print("Training Logistic Regression...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    model = LogisticRegression(max_iter=1000, class_weight={0: class_weights[0], 1: class_weights[1]})
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    print("Training Random Forest...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    model = RandomForestClassifier(n_estimators=100, class_weight={0: class_weights[0], 1: class_weights[1]})
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    print("Training XGBoost...")
    model = XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    # Load preprocessed data
    X_train, y_train, X_test, y_test = load_preprocessed_data('data/processed_train', 'data/processed_test')

    # Resample the data to balance the classes
    X_train_resampled, y_train_resampled = resample_data(X_train, y_train)

    # Train and evaluate Logistic Regression
    log_reg_model = train_logistic_regression(X_train_resampled, y_train_resampled)
    print("Logistic Regression:")
    evaluate_model(log_reg_model, X_test, y_test)

    # Train and evaluate Random Forest
    rf_model = train_random_forest(X_train_resampled, y_train_resampled)
    print("Random Forest:")
    evaluate_model(rf_model, X_test, y_test)

    # Train and evaluate XGBoost
    xgb_model = train_xgboost(X_train_resampled, y_train_resampled)
    print("XGBoost:")
    evaluate_model(xgb_model, X_test, y_test)
