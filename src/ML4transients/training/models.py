import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, input_shape=(30, 30, 1), num_classes=2, num_conv_blocks=2,
                 filters_1=32, filters_2=64, filters_3=128, filters_4=256,
                 dropout_1=0.25, dropout_2=0.25, dropout_3=0.5, units=128):
        super(CustomCNN, self).__init__()
        
        self.num_conv_blocks = num_conv_blocks
        filters = [filters_1, filters_2, filters_3, filters_4]
        dropouts = [dropout_1, dropout_2, dropout_1, dropout_2]  # Reuse dropout values
        
        # Build convolutional blocks dynamically
        self.conv_blocks = nn.ModuleList()
        in_channels = 1
        
        for i in range(num_conv_blocks):
            out_channels = filters[i]
            dropout_rate = dropouts[i]
            
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_rate)
            )
            self.conv_blocks.append(block)
            in_channels = out_channels
        
        # Calculate flattened size after all convolutions
        # Each conv block reduces spatial dimensions by 2 (due to MaxPool2d(2,2))
        spatial_size = 30  # Initial size
        for _ in range(num_conv_blocks):
            spatial_size = spatial_size // 2
        
        self.flattened_size = in_channels * spatial_size * spatial_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, units)
        self.dropout_fc = nn.Dropout(dropout_3)
        self.fc2 = nn.Linear(units, 1)  # Single output for binary classification
        
    def forward(self, x):
        # Apply all convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Flatten and apply fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x