import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.models import create_logistic_regression_model, create_random_forest_model, create_deep_nn_model
import torch
import torch.optim as optim
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_logistic_regression(X_train, y_train, config):
    model = create_logistic_regression_model(config['model']['logistic_regression']['class_weight'])
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, config):
    model = create_random_forest_model(config['model']['random_forest']['class_weight'])
    model.fit(X_train, y_train)
    return model

def train_deep_nn(X_train, y_train, config):
    input_dim = X_train.shape[1]
    num_classes = 1
    model = create_deep_nn_model(input_dim, num_classes, config['model']['deep_nn']['dropout'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['model']['deep_nn']['learning_rate'])

    num_epochs = config['model']['deep_nn']['epochs']
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
    from src.data_preprocessing import preprocess_data

    config = load_config()
    X_train, X_test, y_train, y_test = preprocess_data(config)
    
    # Train models
    log_reg_model = train_logistic_regression(X_train, y_train, config)
    rf_model = train_random_forest(X_train, y_train, config)
    deep_nn_model = train_deep_nn(X_train, y_train, config)
    
    # Save models
    torch.save(deep_nn_model.state_dict(), 'deep_nn_model.pth')
    import joblib
    joblib.dump(log_reg_model, 'log_reg_model.pkl')
    joblib.dump(rf_model, 'rf_model.pkl')
