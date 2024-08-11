import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# Define the Deep Neural Network model
class DeepNN(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

def create_logistic_regression_model(class_weight='balanced'):
    return LogisticRegression(class_weight=class_weight, random_state=42)

def create_random_forest_model(class_weight='balanced'):
    return RandomForestClassifier(class_weight=class_weight, random_state=42)

def create_deep_nn_model(input_dim, num_classes, dropout=0.5):
    return DeepNN(input_dim, num_classes, dropout)
