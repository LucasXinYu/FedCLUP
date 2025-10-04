import torch
from torch import nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        # Single linear layer for logistic regression
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Flatten the input if necessary (i.e., if input is an image)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten batch of inputs to (batch_size, input_dim)
        # Apply the linear layer
        logits = self.fc(x)
        return logits