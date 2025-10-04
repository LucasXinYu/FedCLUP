import torch
from torchvision import datasets, transforms
import numpy as np
import random
import json
import os
from tqdm import trange
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate MNIST dataset for Federated Learning")
    parser.add_argument('--num_clients', type=int, default=1000, help='Number of clients')
    parser.add_argument('--client_classes', type=int, default=2, help='Number of classes per client')
    parser.add_argument('--train_path', type=str, default='../data/mnist/train/all_data_uniform_train.json', help='Path to save training data')
    parser.add_argument('--test_path', type=str, default='../data/mnist/test/all_data_uniform_test.json', help='Path to save testing data')
    parser.add_argument('--evaluation_path', type=str, default='../data/mnist/test/mnist_test.json', help='Path to save evaluation data')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed') 
    return parser.parse_args()

def generate_mnist_data(args):
    # Setup directory for train/test data
    train_path = args.train_path
    test_path = args.test_path
    evaluation_path = args.evaluation_path

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

    # Convert PyTorch dataset to numpy arrays
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

    # Partition the dataset by class
    mnist_data = {}
    for i in range(10):  # MNIST has 10 classes
        mnist_data[i] = (X_train[y_train == i].tolist(), y_train[y_train == i].tolist())

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # Number of samples per class per client
    samples_per_class_per_client = len(X_train) // (args.num_clients * args.client_classes)

    # Assign samples to clients
    for client_id in trange(args.num_clients):
        uname = f'f_{client_id:05d}'
        X_client = []
        y_client = []

        np.random.seed(client_id+args.seed*args.num_clients) # Control the randomness seed
        # Randomly choose client_classes unique classes for each client
        assigned_classes = np.random.choice(10, args.client_classes, replace=False)

        # Flag to indicate if the client can be created
        sufficient_samples = True

        for cls in assigned_classes:
            # Check if there are enough samples left in this class
            X_class, y_class = mnist_data[cls]
            if len(X_class) < samples_per_class_per_client:
                sufficient_samples = False
                break

            # Take samples_per_class_per_client samples from this class and remove them from mnist_data
            X_client += X_class[:samples_per_class_per_client]
            y_client += y_class[:samples_per_class_per_client]

            # Remove the assigned samples from the class
            mnist_data[cls] = (X_class[samples_per_class_per_client:], y_class[samples_per_class_per_client:])

        # If any class did not have enough samples, skip this client
        if not sufficient_samples:
            print(f"Skipping client {uname} due to insufficient samples in a class.")
            continue

        num_samples = len(X_client)
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len

        # Save training data for the client
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X_client[:train_len], 'y': y_client[:train_len]}
        train_data['num_samples'].append(train_len)

        # Save testing data for the client
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_client[train_len:], 'y': y_client[train_len:]}
        test_data['num_samples'].append(test_len)

    # Print summary
    print(train_data['num_samples'])
    print(sum(train_data['num_samples']))

    evaluation_data = {'x': X_test.tolist(), 'y': y_test.tolist()}

    # Save to JSON files
    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)
    with open(evaluation_path, 'w') as outfile:
        json.dump(evaluation_data, outfile)

    print(f"Data successfully saved at {train_path}, {test_path}, {evaluation_path}")


if __name__ == "__main__":
    args = parse_args()
    generate_mnist_data(args)