import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, input_shape=(30, 30, 1), num_classes=2, filters_1=32, filters_2=64, 
                 dropout_1=0.25, dropout_2=0.25, dropout_3=0.5, units=128):
        super(CustomCNN, self).__init__()
        
        # Input shape: (batch_size, 1, 30, 30)
        self.conv1 = nn.Conv2d(1, filters_1, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_1)
        
        self.conv2 = nn.Conv2d(filters_1, filters_2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_2)
        
        # Calculate flattened size after convolutions
        # After two 2x2 pooling operations: 30 -> 15 -> 7 (with floor division)
        self.flattened_size = filters_2 * 7 * 7
        
        self.fc1 = nn.Linear(self.flattened_size, units)
        self.dropout3 = nn.Dropout(dropout_3)
        self.fc2 = nn.Linear(units, 1)  # Single output for binary classification
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x