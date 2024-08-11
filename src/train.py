import sys
import os
import numpy as np
import torch
import torch.optim as optim
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from sklearn.metrics import classification_report

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import create_logistic_regression_model, create_random_forest_model, create_deep_nn_model

# Load the preprocessed data
def load_preprocessed_data(train_path, test_path):
    X_train = np.load(train_path + "_X.npy", allow_pickle=True)
    y_train = np.load(train_path + "_y.npy", allow_pickle=True)
    X_test = np.load(test_path + "_X.npy", allow_pickle=True)
    y_test = np.load(test_path + "_y.npy", allow_pickle=True)
    return X_train, y_train, X_test, y_test

def train_logistic_regression(X_train, y_train):
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def train_deep_nn(X_train, y_train, input_dim, num_epochs=20):
    print("Training Deep Neural Network...")
    model = create_deep_nn_model(input_dim=input_dim, num_classes=1, dropout=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor).squeeze()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model

if __name__ == "__main__":
    # Load preprocessed data
    X_train, y_train, X_test, y_test = load_preprocessed_data('data/processed_train', 'data/processed_test')
    
    # Train models
    log_reg_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    input_dim = X_train.shape[1]  # Number of input features
    deep_nn_model = train_deep_nn(X_train, y_train, input_dim)
    
    # Save models
    torch.save(deep_nn_model.state_dict(), 'deep_nn_model.pth')
    joblib.dump(log_reg_model, 'log_reg_model.pkl')
    joblib.dump(rf_model, 'rf_model.pkl')
    
    # Evaluate the models on the test set
    print("Evaluating Logistic Regression...")
    y_pred_log_reg = log_reg_model.predict(X_test)
    print(classification_report(y_test, y_pred_log_reg))
    
    print("Evaluating Random Forest...")
    y_pred_rf = rf_model.predict(X_test)
    print(classification_report(y_test, y_pred_rf))
    
    print("Evaluating Deep Neural Network...")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    deep_nn_model.eval()
    with torch.no_grad():
        y_pred_nn = deep_nn_model(X_test_tensor).cpu().numpy()
    y_pred_nn = (y_pred_nn > 0.5).astype(int)
    print(classification_report(y_test, y_pred_nn))
