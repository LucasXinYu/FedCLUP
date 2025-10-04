import torch
from torchvision import datasets, transforms
import numpy as np
import random
import json
import os
from tqdm import trange

# Number of clients
num_clients = 1000

# Setup directory for train/test data
train_path = '../data/mnist/train/all_data_uniform_train.json'
test_path = '../data/mnist/test/all_data_uniform_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Load MNIST dataset using PyTorch
transform = transforms.Compose([transforms.ToTensor()])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Convert PyTorch dataset to numpy arrays for easier handling
X_train = mnist_train.data.numpy()
y_train = mnist_train.targets.numpy()
X_test = mnist_test.data.numpy()
y_test = mnist_test.targets.numpy()

# Normalize the data (between 0 and 1)
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# **Flatten the images to 1D arrays** (28 * 28 = 784)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print(f"Train data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Split training data equally for each client
X_users = []
y_users = []

num_samples_per_client = len(X_train) // num_clients
for i in range(num_clients):
    X_users.append(X_train[i * num_samples_per_client: (i + 1) * num_samples_per_client])
    y_users.append(y_train[i * num_samples_per_client: (i + 1) * num_samples_per_client])

# Create data structure
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}

# For each client, split their data into training (90%) and testing (10%)
for i in trange(num_clients):
    uname = f'f_{i:05d}'
    
    combined = list(zip(X_users[i], y_users[i]))
    X_users[i][:], y_users[i][:] = zip(*combined)
    
    # Convert NumPy arrays to lists for JSON serialization
    X_list = np.array(X_users[i]).tolist()
    y_list = np.array(y_users[i]).tolist()
    
    num_samples = len(X_users[i])
    train_len = int(0.9 * num_samples)
    test_len = num_samples - train_len
    
    # Save training data for the client
    train_data['users'].append(uname)
    train_data['user_data'][uname] = {'x': X_list[:train_len], 'y': y_list[:train_len]}
    train_data['num_samples'].append(train_len)
    
    # Save testing data for the client
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X_list[train_len:], 'y': y_list[train_len:]}
    test_data['num_samples'].append(test_len)

# Print summary
print(train_data['num_samples'])
print(sum(train_data['num_samples']))

# Save to JSON files
with open(train_path, 'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print(f"Data successfully saved at {train_path} and {test_path}")