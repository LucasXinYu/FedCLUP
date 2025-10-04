######################
###### PFL algorithm 
######################

# -*- coding:utf-8 -*-
import copy
import numpy as np
import pandas as pd
import os
import random
from sklearn.preprocessing import StandardScaler
import multiprocessing
import matplotlib.pyplot as plt
import pdb
import argparse

def run_federated_learning(args):
    # Load and scale data for clients
    client_data_X, client_data_Y, scalars = load_and_scale_data(args.num_clients, args.input_dim, args.num_classes, args.data_dir)

    # Calculate strong str_convexity and smoothness
    str_convexity, smoothness = calculate_mu_and_L(client_data_X[1])
    print(f'Strong str_convexity: {str_convexity}, Smoothness: {smoothness}')

    np.random.seed(1) ### RANDOMNESS FIXED
    # Initialize global and local models for FedProx, PureLocal, and FedAvg
    global_model = np.random.randn(args.input_dim, args.num_classes)
    client_models = [copy.deepcopy(global_model) for _ in range(args.num_clients)]

    # Check if the first run flag is set or not
    if not args.first_run:
        if args.record_error == 'opt':
            # Load global and local minimizers if not the first run
            if args.method == 'FedProx':
                # Load global minimizer
                global_minimizer = np.load(os.path.join(args.output_data_dir, f'global_minimizer_{args.method}_mu{args.mu}.npy'))

                # Load all client minimizers from a single file
                local_minimizers = np.load(os.path.join(args.output_data_dir, f'local_minimizers_{args.method}_mu{args.mu}.npy'))
                print(f"All client minimizers loaded from 'local_minimizers_{args.method}_mu{args.mu}.npy'")

            else:
                # Load global minimizer
                global_minimizer = np.load(os.path.join(args.output_data_dir, f'global_minimizer_{args.method}.npy'))

                # Load all client minimizers from a single file
                local_minimizers = np.load(os.path.join(args.output_data_dir, f'local_minimizers_{args.method}.npy'))
                print(f"All client minimizers loaded from 'local_minimizers_{args.method}.npy'")
        elif args.record_error == 'stat':
            # Load local/global weights (of different shape, should take transpose)
            local_true_model = np.load(os.path.join(args.data_dir, 'local_true_model.npy'), allow_pickle=True)
            global_true_model = np.load(os.path.join(args.data_dir, 'global_true_model.npy'))
        else:
            raise ValueError("the type of record_error cannot be identified")

        AOR, IOR = [], []
        ATR, ITR = [], []
    # Update the previous models for the next iteration
    prev_global_model = np.copy(global_model)

    mu_record = args.mu
    if_adjusted = False

    # Training loop
    for round_num in range(args.num_rounds):
        print(f"Round {round_num + 1}:")
        # print(client_models[1]) ## TEST
        if round_num >= int(args.num_rounds*0.5) and args.adapt_mu == True and not if_adjusted:
            total_distance = 0.0
            # Calculate the L2 distance for each local model to the global model
            for local_model in client_models:
                l2_distance = np.linalg.norm(local_model - global_model)
                total_distance += l2_distance

            # Return the average distance
            R_est = total_distance / args.num_clients
            args.mu = (1/(200*R_est**2) if R_est >= 1/np.sqrt(200) else 1/np.sqrt(200*R_est**2)) * 5 # Hardcode local size & rho
            print(f"Round {round_num + 1}: START TO ADAPT, mu = {args.mu}, R = {R_est}.")
            args.lr_global = 1/args.mu + 1/9.13
            if_adjusted = True

        # Sample clients
        num_sampled_clients = max(int(args.sampling_rate * args.num_clients), 1)
        sampled_clients = random.sample(range(args.num_clients), num_sampled_clients)

        # Prepare data for parallel processing
        sampled_client_data = [(client_idx, client_data_X[client_idx], client_data_Y[client_idx],
                        client_models[client_idx]) for client_idx in sampled_clients]

        # Use multiprocessing to parallelize local training
        with multiprocessing.Pool() as pool:
            results = pool.starmap(local_training_update, [(data, global_model, round_num, args) for data in sampled_client_data])

        # Collect the results and update the client models
        for result in results:
            client_idx, client_model = result
            client_models[client_idx] = client_model

        # Global model aggregation
        if args.method == "FedAvg" or "LocalTrain":
            global_model = sum(client_models) / len(client_models)
        elif args.method == "FedProx":
            global_model = aggregation(client_models, global_model, sampled_clients, args.mu, args.lr_global)
        else:
            raise ValueError(f"Cannot identify the method {args.method}")

        if not args.first_run:
            if args.record_error == 'opt':
                # Calculate AOR and IOR
                AOR_val = np.linalg.norm(global_model - global_minimizer)
                
                if args.method == "FedAvg":
                    IOR_val = np.linalg.norm(client_models[0] - global_minimizer)
                else:
                    IOR_val = np.linalg.norm(client_models[0] - local_minimizers[0])

                # Append AOR and IOR for each algorithm
                AOR.append(AOR_val)
                IOR.append(IOR_val)
                print(f"{args.method} AOR: {AOR[-1]}, IOR: {IOR[-1]}")
            elif args.record_error == 'stat':
                # For ATR: Calculate Total Error for Global model (global_model - global_true_model)
                # Assuming no scaling for the global model
                ATR_val = np.linalg.norm(global_model - global_true_model.T)

                # For ITR: Calculate Total Error for each Local model (client_models - local_true_model)
                # Apply scaling factors to adjust for scaling applied during data loading
                ITR_val = np.mean([
                    np.linalg.norm(client_model / scalars[client_id][:, None] - true_model.T)
                    for client_id, (client_model, true_model) in enumerate(zip(client_models, local_true_model))
                ])

                # Append ATR and ITR for each algorithm
                ATR.append(ATR_val)
                ITR.append(ITR_val)
                print(f"{args.method} ATR: {ATR[-1]}, ITR: {ITR[-1]}")

        # Calculate the norm of the difference between the current and previous models
        diff = np.linalg.norm(global_model - prev_global_model)

        # Print the norm of the differences
        print(f"{args.method} Norm of model difference: {diff}")   
        # Update the previous models for the next iteration
        prev_global_model = np.copy(global_model)  

        if np.abs(diff) < args.stopping_threshold:
            print(f"Converged at round {round_num + 1}. Stopping early.")
            break      

    if args.first_run:
        os.makedirs(args.output_data_dir, exist_ok = True)
        # Save global and local minimizers after convergence
        if args.method == "FedProx":
            # Save global minimizer
            np.save(os.path.join(args.output_data_dir, f'global_minimizer_{args.method}_mu{args.mu}.npy'), global_model)
            print("Global Minimizer:", global_model)

            # Save all client minimizers in a single file
            local_minimizers_array = np.array(client_models)
            np.save(os.path.join(args.output_data_dir, f'local_minimizers_{args.method}_mu{args.mu}.npy'), local_minimizers_array)
            print(f"All client minimizers saved to 'local_minimizers_{args.method}_mu{args.mu}.npy'")

        else:
            # Save global minimizer
            np.save(os.path.join(args.output_data_dir, f'global_minimizer_{args.method}.npy'), global_model)
            print("Global Minimizer:", global_model)

            # Save all client minimizers in a single file
            local_minimizers_array = np.array(client_models)
            np.save(os.path.join(args.output_data_dir, f'local_minimizers_{args.method}.npy'), local_minimizers_array)
            print(f"All client minimizers saved to 'local_minimizers_{args.method}.npy'")
    else:
        if args.record_error == 'opt':
            # Save the AOR and IOR values for different methods and mu for FedProx
            if args.method == 'FedProx':
                aor_file = os.path.join(args.output_data_dir, 
                                        f'AOR_{args.method}_mu{args.mu}_comm{args.num_rounds}_epochs{args.local_epochs}.npy')
                ior_file = os.path.join(args.output_data_dir, 
                                        f'IOR_{args.method}_mu{args.mu}_comm{args.num_rounds}_epochs{args.local_epochs}.npy')
            else:
                aor_file = os.path.join(args.output_data_dir, 
                                        f'AOR_{args.method}_comm{args.num_rounds}_epochs{args.local_epochs}.npy')
                ior_file = os.path.join(args.output_data_dir, 
                                        f'IOR_{args.method}_comm{args.num_rounds}_epochs{args.local_epochs}.npy')

            # Convert AOR and IOR lists to NumPy arrays and save them
            np.save(aor_file, np.array(AOR))
            np.save(ior_file, np.array(IOR))
            print(f"AOR, IOR saved to {args.output_data_dir}")
        elif args.record_error == 'stat':
            # Save the ATR and ITR values for different methods and mu for FedProx
            if args.R is None:
                if args.method == 'FedProx':
                    # Include mu in the filename for FedProx
                    atr_file = os.path.join(args.output_data_dir, f'ATR_{args.method}_mu{mu_record}.npy')
                    itr_file = os.path.join(args.output_data_dir, f'ITR_{args.method}_mu{mu_record}.npy')
                else:
                    atr_file = os.path.join(args.output_data_dir, f'ATR_{args.method}.npy')
                    itr_file = os.path.join(args.output_data_dir, f'ITR_{args.method}.npy')
            else:
                if args.method == 'FedProx':
                    # Include mu in the filename for FedProx
                    atr_file = os.path.join(args.output_data_dir, f'ATR_{args.method}_mu{mu_record}_R{args.R}.npy')
                    itr_file = os.path.join(args.output_data_dir, f'ITR_{args.method}_mu{mu_record}_R{args.R}.npy')
                else:
                    atr_file = os.path.join(args.output_data_dir, f'ATR_{args.method}_R{args.R}.npy')
                    itr_file = os.path.join(args.output_data_dir, f'ITR_{args.method}_R{args.R}.npy')

            # Convert ATR and ITR lists to NumPy arrays and save them
            os.makedirs(args.output_data_dir, exist_ok = True)
            np.save(atr_file, np.array(ATR))
            np.save(itr_file, np.array(ITR))
            print(f"ATR, ITR saved to {args.output_data_dir}")

def local_training_update(client_data, global_model, round_num, args):
    client_idx, X, Y, client_model = client_data

    if args.adapt_local_epochs:
        local_epochs = round_num
    else:
        local_epochs = args.local_epochs

    if args.method == 'FedAvg':
        client_model = np.copy(global_model)
        for _ in range(local_epochs):
            # FedAvg update
            logits_fedavg = X @ client_model
            probs_fedavg = softmax(logits_fedavg)
            grad_fedavg = X.T @ (probs_fedavg - Y) / len(Y)
            client_model -= args.lr_local * grad_fedavg
    elif args.method == 'FedProx':
        for _ in range(local_epochs):
            # FedProx update
            logits = X @ client_model
            probs = softmax(logits)
            grad_fedprox = X.T @ (probs - Y) / len(Y) + args.mu * (client_model - global_model)
            client_model -= args.lr_local * grad_fedprox
    elif args.method == 'LocalTrain':
        for _ in range(local_epochs):
            # Pure Local Training update
            logits_local = X @ client_model
            probs_local = softmax(logits_local)
            grad_local = X.T @ (probs_local - Y) / len(Y)
            client_model -= args.lr_local * grad_local
        # ## TEST:
        # if client_idx == 1:
        #     print("The response for client 1 is", Y)

    return (client_idx, client_model)

def load_and_scale_data(num_clients, model_dim, num_classes, data_dir, scale=True):
    """
    Load and scale data for each client, and return the scaling factors.
    
    Args:
    - num_clients: Number of clients.
    - model_dim: Dimension of the feature vector (input).
    - num_classes: Number of output classes.
    - data_dir: Directory where client data is stored.
    - scale: Whether to apply scaling.
    
    Returns:
    - X: Dictionary containing the feature matrix for each client.
    - Y: Dictionary containing the one-hot encoded labels for each client.
    - scalars: Dictionary containing the scaling factor for each client (if scaling is applied).
    """
    X = {}
    Y = {}
    scalars = {}

    for client_id in range(num_clients):
        # Construct the file path for each client's data
        file_path = os.path.join(data_dir, f'{client_id+1}_client_data.csv')
        data = pd.read_csv(file_path)

        if scale:
            # Apply scaling and store the scalar (standard deviation) for each client
            scaler = StandardScaler(with_mean=False)
            data.iloc[:, :model_dim] = scaler.fit_transform(data.iloc[:, :model_dim])
            scalars[client_id] = scaler.scale_  # Store scaling factor
        else:
            scalars[client_id] = np.ones(model_dim)  # No scaling, scalar is 1

        # Store the scaled features and target labels in dictionaries
        X[client_id] = data.iloc[:, :model_dim].values.astype(np.float64)
        labels = np.array(data.iloc[:, -1].values.astype(np.int16))  # Last column contains labels
        Y[client_id] = np.eye(num_classes)[labels.reshape(-1)]  # One-hot encoded labels

    return X, Y, scalars


def calculate_mu_and_L(X):
    """
    Calculate the strong convexity parameter (mu) and smoothness constant (L).

    Args:
    - X: Feature matrix for a client.

    Returns:
    - mu: Smallest eigenvalue of X^T * X / n (strong convexity parameter).
    - L: Largest eigenvalue of X^T * X / n (smoothness constant).
    """
    num_samples = X.shape[0]  # Number of samples (rows)
    XtX = 1/4 * np.dot(X.T, X).astype(np.float64) / num_samples  # Compute X^T * X / n

    # Compute the eigenvalues of X^T * X / n
    eigenvalues = np.linalg.eigvalsh(XtX)

    mu = np.min(eigenvalues)  # Strong convexity parameter
    L = np.max(eigenvalues)   # Smoothness constant

    return mu, L

def aggregation(local_models, global_model, client_indices, proximal_term, global_lr):
    """
    Aggregate client models to update the global model using FedProx.

    Args:
    - local_models: List of local models from clients.
    - global_model: Current global model.
    - client_indices: List of selected client indices.
    - proximal_term: Proximal regularization term (mu).
    - global_lr: Global learning rate.

    Returns:
    - Updated global model after aggregation.
    """
    adjustment = 0
    for client_idx in client_indices:
        adjustment += proximal_term * (global_model - local_models[client_idx])  / len(client_indices) # sum (wi - wg) /m
    global_model -= global_lr * adjustment

    return global_model


def softmax(logits):
    """
    Compute the softmax of a matrix of logits.

    Args:
    - logits: Input matrix (n_samples, n_classes).

    Returns:
    - Softmax probabilities.
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def args_parser_fl():
    """
    Parse command-line arguments for Federated Learning (FL) experiments.

    Returns:
    - args: Parsed arguments containing various hyperparameters and configurations.
    """
    parser = argparse.ArgumentParser(description="Federated Learning Argument Parser")

    # key arguments
    parser.add_argument('--method', type=str, required=True, help='Method for testing.')
    parser.add_argument('--record_error', type=str, choices=['stat', 'opt'], help='Which type of error to record.')
    parser.add_argument('--data_dir', type=str, required=True, help='Input Data directory.')
    parser.add_argument('--output_data_dir', type=str, required=True, help='Output Data directory.')
    parser.add_argument('--first_run', action='store_true', help='Run the algorithm once to store the optimal points for calculating optimization error.')

    # Data Configuration
    parser.add_argument('--num_clients', type=int, default=100, help='Total number of clients')
    parser.add_argument('--input_dim', type=int, default=60, help='Input dimension of feature vectors')
    parser.add_argument('--local_batch_size', type=int, default=50, help='Local batch size for client training')
    parser.add_argument('--sampling_rate', type=float, default=1, help='Fraction of clients to sample per round (C)')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes')

    # Training Iteration Configuration
    parser.add_argument('--num_rounds', type=int, default=100, help='Number of communication rounds (r)')
    parser.add_argument('--local_epochs', type=int, default=1000, help='Number of local training epochs per round (E)')
    parser.add_argument('--adapt_local_epochs', action='store_true', help='Set the number of local update epochs to be proportional to communication rounds')
    parser.add_argument('--adapt_mu', action='store_true', help='Set the regularization strength mu according to the theoretical setup')
    
    # Hyperparameter Configuration
    parser.add_argument('--mu', type=float, default=5, help='Proximal term regularization (mu) for FedProx')
    parser.add_argument('--stopping_threshold', type=float, default=-1, help='Stopping rule for the outer loop based on the precision threshold')
    parser.add_argument('--R', type=float, default=None, help='An optional parameter specifying R, it will be included in the file name') 

    # Learning Rates
    parser.add_argument('--lr_local', type=float, default=0.1, help='Local learning rate for client updates')
    parser.add_argument('--lr_global', type=float, default=0.1, help='Global learning rate for server updates (gradient descent)')
 
    # Parse and return arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Parse arguments and initialize
    args = args_parser_fl()

    # Run federated learning with or without AOR/IOR assessment
    run_federated_learning(args)