import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN, self).__init__()
        # Assuming input_dim is the flattened size of MNIST images (28*28 = 784)
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Define CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        # The output of the conv2 layer will be of size [batch_size, 64, 7, 7] after two pooling layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

    def forward(self, x):
        # Check if the input is flattened; if so, reshape it to the expected dimensions
        if x.dim() == 2:  # Input is [batch_size, 784]
            x = x.view(-1, 1, 28, 28)  # Reshape to [batch_size, 1, 28, 28]

        # Pass input through the first convolutional layer + ReLU + MaxPool
        x = self.pool(torch.relu(self.conv1(x)))

        # Pass through the second convolutional layer + ReLU + MaxPool
        x = self.pool(torch.relu(self.conv2(x)))

        # Flatten the output from the conv layers to feed into the fully connected layers
        x = x.view(-1, 64 * 7 * 7)  # Flatten to [batch_size, 64*7*7]

        # Pass through fully connected layers
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)  # Output logits

        return logits