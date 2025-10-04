# visualize_stat_error.py

import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pdb
import re

# Small constant to avoid log(0)
epsilon = 1e-15

def load_stat_results(method, R, mu=None, output_data_dir=None):
    """
    Load ATR and ITR results for a given method (and mu for FedProx) for a specific value of R.
    
    Args:
        method (str): The name of the method (e.g., 'FedAvg', 'FedProx', 'LocalTrain').
        R (float): The value of R used during the simulation.
        mu (float, optional): The value of mu for FedProx.
        output_data_dir (str): Directory where ATR and ITR files are saved.
    
    Returns:
        atr (float): Final value of ATR.
        itr (float): Final value of ITR.
    """
    if method == 'FedProx' and mu is not None:
        atr_file = os.path.join(output_data_dir, f'ATR_{method}_mu{mu}_R{R}.npy')
        itr_file = os.path.join(output_data_dir, f'ITR_{method}_mu{mu}_R{R}.npy')
    else:
        atr_file = os.path.join(output_data_dir, f'ATR_{method}_R{R}.npy')
        itr_file = os.path.join(output_data_dir, f'ITR_{method}_R{R}.npy')

    # Load the full arrays and extract the last value (final error)
    atr = np.load(atr_file)[-1]
    itr = np.load(itr_file)[-1]

    return atr, itr

def plot_stat_error(atr_dict, itr_dict, output_dir, R_values):
    """
    Plot ATR and ITR as a function of R for different methods and lambda values for FedProx in a compact format.
    Only plot half of the data points (i.e., 1st, 3rd, 5th, etc.).
    
    Args:
        atr_dict (dict): Dictionary with method names (and lambda for FedProx) as keys and final ATR values as values.
        itr_dict (dict): Dictionary with method names (and lambda for FedProx) as keys and final ITR values as values.
        output_dir (str): Directory where to save the plots.
        R_values (list): List of R values used during simulations.
    """
    # Set the figure size to 4x4
    fig_size = (4, 4)  # Width x Height
    marker_size = 4  # Adjust marker size for better visibility
    font_size = 10  # Font size for axis labels and legend
    legend_font_size = 8  # Font size for the legend
    default_alpha = 1.0  # Default transparency for methods without specific alpha
    grid_alpha = 0.3  # Transparency for the grid

    # Get colors from the 'tab10' color palette
    tab10_colors = plt.get_cmap('tab10').colors

    # Define method styles with the requested color changes and symbols
    method_styles = {
        'FedAvg': {'color': tab10_colors[0], 'linestyle': '--', 'marker': 'o', 'label': 'FedAvg', 'alpha': 0.9},  # Blue, with transparency
        'LocalTrain': {'color': tab10_colors[7], 'linestyle': '--', 'marker': 's', 'label': 'LocalTrain', 'alpha': default_alpha}  # Grey, no transparency
    }

    # Extract and sort the lambda (mu) values from FedProx methods
    fedprox_methods = [method for method in atr_dict.keys() if 'FedProx' in method]
    mu_values = []

    for method in fedprox_methods:
        match = re.search(r'_mu([0-9.]+)', method)
        if match:
            mu_values.append((method, float(match.group(1))))

    # Sort the FedProx methods by their lambda (mu) values
    mu_values.sort(key=lambda x: x[1])
    mu_labels = ['small lambda', 'medium lambda', 'large lambda']
    fedprox_colors = [tab10_colors[2], tab10_colors[1], tab10_colors[3]]  # Green, Orange, Red

    # Assign markers and transparency to FedProx methods
    fedprox_markers = ['D', '*', 'x']  # Diamond, Star, Multiply

    # Assign styles to the sorted FedProx methods
    for (method, _), label, color, marker in zip(mu_values, mu_labels, fedprox_colors, fedprox_markers):
        alpha = 0.9 if label == 'large lambda' else default_alpha  # Transparency only for large lambda
        method_styles[method] = {
            'color': color, 'linestyle': '-', 'marker': marker, 'label': f'FedProx ({label})', 'alpha': alpha
        }

    # Get every second R value and corresponding ATR/ITR data
    R_values_half = R_values[::2]  # Take every second R value (first, third, fifth, etc.)

    # Plot ATR (Global Statistical Error) as a function of R
    plt.figure(figsize=fig_size)
    for method, atr in atr_dict.items():
        style = method_styles.get(method, {})
        plt.plot(R_values_half, atr[::2], color=style['color'], linestyle=style['linestyle'],
                 marker=style['marker'], markersize=marker_size, label=style['label'], alpha=style['alpha'])
    
    plt.xlabel('Statistical Heterogeneity', fontsize=font_size)
    plt.ylabel('Global Statistical Error', fontsize=font_size)

    # Remove top and right borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Position the legend below the plot
    plt.legend(fontsize=legend_font_size, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)

    # Add a grid with transparency
    plt.grid(True, alpha=grid_alpha)

    plt.tight_layout()  # Ensure the plot fits well in a small space
    plt.savefig(os.path.join(output_dir, 'GSE_vs_R_half_points.png'), bbox_inches='tight')
    plt.close()

    # Plot ITR (Local Statistical Error) as a function of R
    plt.figure(figsize=fig_size)
    for method, itr in itr_dict.items():
        style = method_styles.get(method, {})
        plt.plot(R_values_half, itr[::2], color=style['color'], linestyle=style['linestyle'],
                 marker=style['marker'], markersize=marker_size, label=style['label'], alpha=style['alpha'])

    plt.xlabel('Statistical Heterogeneity (R)', fontsize=font_size)
    plt.ylabel('Local Statistical Error', fontsize=font_size)

    # Remove top and right borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Position the legend below the plot
    plt.legend(fontsize=legend_font_size, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)

    # Add a grid with transparency
    plt.grid(True, alpha=grid_alpha)

    plt.tight_layout()  # Ensure the plot fits well in a small space
    plt.savefig(os.path.join(output_dir, 'LSE_vs_R_half_points.png'), bbox_inches='tight')
    plt.close()

# Main function to handle arguments and run the visualization
def main():
    parser = argparse.ArgumentParser(description='Visualize ATR and ITR for different methods over R.')

    # Arguments for the script
    parser.add_argument('--output_data_dir', type=str, required=True, help='Directory where the results are saved.')
    parser.add_argument('--R_values', type=float, nargs='+', required=True, help='List of R values used during simulations.')
    parser.add_argument('--methods', type=str, nargs='+', required=True, help='List of methods to visualize (e.g., FedAvg, LocalTrain, FedProx).')
    parser.add_argument('--mu_values', type=float, nargs='*', help='List of mu values for FedProx.')

    args = parser.parse_args()

    # Load ATR and ITR results for each method and R
    atr_dict = {}
    itr_dict = {}

    # Load results for FedAvg and LocalTrain
    for method in args.methods:
        if method != 'FedProx':
            atr_list = []
            itr_list = []
            for R in args.R_values:
                atr, itr = load_stat_results(method, R, output_data_dir=args.output_data_dir)
                atr_list.append(atr)
                itr_list.append(itr)
            atr_dict[method] = atr_list
            itr_dict[method] = itr_list

    # Load results for FedProx with different mu values
    if 'FedProx' in args.methods:
        for mu in args.mu_values:
            atr_list = []
            itr_list = []
            for R in args.R_values:
                atr, itr = load_stat_results('FedProx', R, mu=mu, output_data_dir=args.output_data_dir)
                atr_list.append(atr)
                itr_list.append(itr)
            atr_dict[f'FedProx_mu{mu}'] = atr_list
            itr_dict[f'FedProx_mu{mu}'] = itr_list
    
    # Plot and save the results
    plot_stat_error(atr_dict, itr_dict, args.output_data_dir, args.R_values)

if __name__ == "__main__":
    main()