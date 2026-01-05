import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, input_shape=(30, 30, 1), num_classes=2, num_conv_blocks=2,
                 filters_1=32, filters_2=64, filters_3=128, filters_4=256,
                 dropout_1=0.25, dropout_2=0.25, dropout_3=0.5, units=128,
                 in_channels=None):
        """
        Convolutional Neural Network for image classification.
        
        Parameters
        ----------
        input_shape : tuple
            Shape of input images (height, width, channels)
        num_classes : int
            Number of output classes
        num_conv_blocks : int
            Number of convolutional blocks
        filters_1, filters_2, filters_3, filters_4 : int
            Number of filters for each conv block
        dropout_1, dropout_2, dropout_3 : float
            Dropout rates
        units : int
            Number of units in fully connected layer
        in_channels : int, optional
            Number of input channels. If None, inferred from input_shape.
            Use this to override for multi-channel inputs (e.g., diff + coadd)
        """
        super(CustomCNN, self).__init__()
        
        self.num_conv_blocks = num_conv_blocks
        filters = [filters_1, filters_2, filters_3, filters_4]
        dropouts = [dropout_1, dropout_2, dropout_1, dropout_2]  # Reuse dropout values
        
        # Determine input channels
        if in_channels is not None:
            # Explicit channel count provided (for multi-channel inputs)
            initial_channels = in_channels
        elif len(input_shape) == 3:
            # Extract from input_shape (height, width, channels)
            initial_channels = input_shape[2]
        else:
            # Default to 1 channel
            initial_channels = 1
        
        print(f"Building CNN with {initial_channels} input channels")
        
        # Build convolutional blocks dynamically
        self.conv_blocks = nn.ModuleList()
        in_ch = initial_channels
        
        for i in range(num_conv_blocks):
            out_channels = filters[i]
            dropout_rate = dropouts[i]
            
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_rate)
            )
            self.conv_blocks.append(block)
            in_ch = out_channels
        
        # Calculate flattened size after all convolutions
        # Each conv block reduces spatial dimensions by 2 (due to MaxPool2d(2,2))
        spatial_size = input_shape[0] if isinstance(input_shape, (list, tuple)) else 30  # Initial size
        for _ in range(num_conv_blocks):
            spatial_size = spatial_size // 2
        
        self.flattened_size = in_ch * spatial_size * spatial_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, units)
        self.dropout_fc = nn.Dropout(dropout_3)
        self.fc2 = nn.Linear(units, 1)  # Single output for binary classification
        
    def enable_mc_dropout(self):
        """
        Enable Monte Carlo Dropout for uncertainty estimation during inference.
        
        This method keeps dropout layers active during inference while maintaining
        batch normalization layers in eval mode. This allows for uncertainty 
        quantification by running multiple forward passes with different dropout masks.
        
        Note: The model must have been trained with dropout for MC Dropout to be effective.
        """
        self.eval()  # First set everything to eval mode
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()  # Re-enable only dropout layers
    
    def disable_mc_dropout(self):
        """
        Restore normal evaluation mode (disable all dropout).
        """
        self.eval()
    
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