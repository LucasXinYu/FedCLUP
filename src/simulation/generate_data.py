import numpy as np
import os
import pandas as pd
import argparse

def generate_device_data(R, num_samples, x_dim, w_dim, b_dim, cov_structure):
    """
    Generates data for a single device based on the given parameters.
    
    Args:
        R: Controls the variance of the weight matrix and bias for each device.
        num_samples: Number of data samples per device.
        x_dim: Dimension of the input feature vector.
        w_dim: Dimension of the weight matrix W.
        b_dim: Dimension of the bias term (output dimension).
    
    Returns:
        X_k: Input feature matrix for the device (num_samples, x_dim).
        Y_k: Corresponding labels for the input samples (num_samples,).
        W_k: Weight matrix for the device (w_dim).
    """

    # Device-specific weight and bias generation
    W_k = np.random.normal(0, R, size=w_dim)  # Weight matrix for device
    # Generate input features X with heterogeneity controlled by beta
    v_k = np.random.normal(0, 1, size=x_dim)  # Variance in feature space

    if cov_structure == "toeplitz":
        # Covariance matrix Σ is diagonal with entries decreasing as j^(-1.2)
        Sigma = np.diag([j**(-1.2) for j in range(1, x_dim + 1)])
    elif cov_structure == "identical":
        raise ValueError
    else:
        raise ValueError
    # Generate the input feature matrix X_k from a multivariate normal distribution
    X_k = np.random.multivariate_normal(v_k, Sigma, size=num_samples) # TEST
    
    # Calculate logits and labels
    logits = np.dot(X_k, W_k.T) # Compute logits
    Y_k = np.argmax(np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True), axis=1)  # Labels via argmax of softmax
    
    return X_k, Y_k, W_k

def main(args):
    # Create folder to store data
    os.makedirs(args.output_dir, exist_ok=True)  # Ensure the directory exists
    
    # List to store W_k matrices for all devices
    W_k_list = []
    
    # Generate data for each device
    for device in range(args.num_devices):
        # Generate data for the current device
        np.random.seed(device+args.seed*args.num_devices) # Control the randomness seed
        X_k, Y_k, W_k = generate_device_data(args.R, args.n_samples, args.x_dim, (args.b_dim, args.x_dim), args.b_dim, args.cov_structure)
        
        # Combine features and labels into one matrix
        data = np.hstack((X_k, Y_k.reshape(-1, 1)))
        
        # Save data to CSV file for each device
        file_name = os.path.join(args.output_dir, f"{device+1}_client_data.csv")
        df = pd.DataFrame(data)
        df.to_csv(file_name, index=False, header=False)
        
        # Print data shapes for debugging
        print(f"Device {device+1} - X_k shape: {X_k.shape}, Y_k shape: {Y_k.shape}, Data shape: {df.shape}")
        
        # Store W_k for this device
        W_k_list.append(W_k)
    
    # Save the local client weights
    local_weights_file = os.path.join(args.output_dir, 'local_true_model.npy')
    np.save(local_weights_file, W_k_list)
    print(f"Local client weights saved to {local_weights_file}")
    
    # Save the global (average) weight matrix
    global_weights_file = os.path.join(args.output_dir, "global_true_model.npy")
    W_k_avg = np.mean(np.array(W_k_list), axis=0)
    np.save(global_weights_file, W_k_avg)
    print(f"Average W_k saved to {global_weights_file}")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Generate client data for federated learning simulation.")
    
    parser.add_argument('--R', type=float, default=0.5, help="Controls variance of the W_k and b_k for different devices.")
    parser.add_argument('--num_devices', type=int, default=10, help="Number of devices (clients) participating in the simulation.")
    parser.add_argument('--n_samples', type=int, default=100, help="Number of samples per device.")
    parser.add_argument('--x_dim', type=int, default=6, help="Dimension of input feature vectors.")
    parser.add_argument('--b_dim', type=int, default=3, help="Dimension of bias term (output dimension).")
    parser.add_argument('--cov_structure', type=str, default="toeplitz", help="Covariance structure of the covariate.")
    parser.add_argument('--output_dir', type=str, default="../data/fedprox_syndata/test", help="Folder name to store the data.")
    parser.add_argument('--seed', type=int, default=1, help='Random Seed') 
   

    args = parser.parse_args()
    
    # Generate the data
    main(args)