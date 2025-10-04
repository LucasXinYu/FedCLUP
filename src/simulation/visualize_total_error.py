import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pdb
import re

# Small constant to avoid log(0)
epsilon = 1e-15

def load_total_results(method, R, mu=None, output_data_dir=None):
    """
    Load ATR and ITR results for a given method (and mu for FedProx) for a specific value of R.
    
    Args:
        method (str): The name of the method (e.g., 'FedAvg', 'FedProx', 'LocalTrain').
        R (float): The value of R used during the simulation.
        mu (float, optional): The value of mu for FedProx.
        output_data_dir (str): Directory where ATR and ITR files are saved.
    
    Returns:
        atr (float): Full sequence of ATR.
        itr (float): Full sequence of ITR.
    """
    if method == 'FedProx' and mu is not None:
        atr_file = os.path.join(output_data_dir, f'ATR_{method}_mu{mu}_R{R}.npy')
        itr_file = os.path.join(output_data_dir, f'ITR_{method}_mu{mu}_R{R}.npy')
    else:
        atr_file = os.path.join(output_data_dir, f'ATR_{method}_R{R}.npy')
        itr_file = os.path.join(output_data_dir, f'ITR_{method}_R{R}.npy')

    # Load the full arrays and extract the full sequence
    atr = np.load(atr_file)
    itr = np.load(itr_file)

    return atr, itr


def plot_total_error(atr_dict, itr_dict, output_dir, R_value, local_epochs):
    """
    Plot ATR and ITR as a function of communication rounds for different methods and lambda values for FedProx.
    
    Args:
        atr_dict (dict): Dictionary with method names (and lambda for FedProx) as keys and final ATR values as values.
        itr_dict (dict): Dictionary with method names (and lambda for FedProx) as keys and final ITR values as values.
        output_dir (str): Directory where to save the plots.
    """
    # Set the figure size and marker size
    fig_size = (4, 4)
    marker_size = 4
    font_size = 10
    legend_font_size = 8
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

    # Plot ATR (Global Total Error) as a function of communication rounds
    plt.figure(figsize=fig_size)
    for method, atr in atr_dict.items():
        style = method_styles.get(method, {})
        plt.plot(range(1, len(atr)+1), atr, color=style['color'], linestyle=style['linestyle'],
                 label=style['label'], alpha=style['alpha'])
    
    plt.xlabel('# Communication Rounds', fontsize=font_size)
    plt.ylabel('Global Total Error', fontsize=font_size)

    # Remove top and right borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Position the legend below the plot
    plt.legend(fontsize=legend_font_size, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)

    # Add a grid with transparency
    plt.grid(True, alpha=grid_alpha)

    plt.tight_layout()  # Ensure the plot fits well in a small space
    plt.savefig(os.path.join(output_dir, f'Global_Total_Error_vs_Comm_Rounds_R{R_value}_epochs{local_epochs}.png'), bbox_inches='tight')
    plt.close()

    # Plot ITR (Local Total Error) as a function of communication rounds
    plt.figure(figsize=fig_size)
    for method, itr in itr_dict.items():
        style = method_styles.get(method, {})
        plt.plot(range(1, len(atr)+1), itr, color=style['color'], linestyle=style['linestyle'],
                 label=style['label'], alpha=style['alpha'])

    plt.xlabel('Communication Rounds', fontsize=font_size)
    plt.ylabel('Local Total Error', fontsize=font_size)

    # Remove top and right borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Position the legend below the plot
    plt.legend(fontsize=legend_font_size, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)

    # Add a grid with transparency
    plt.grid(True, alpha=grid_alpha)

    plt.tight_layout()  # Ensure the plot fits well in a small space
    plt.savefig(os.path.join(output_dir, f'Local_Total_Error_vs_Comm_Rounds_R{R_value}_epochs{local_epochs}.png'), bbox_inches='tight')
    plt.close()

# Main function to handle arguments and run the visualization
def main():
    parser = argparse.ArgumentParser(description='Visualize ATR and ITR for different methods over Communication Rounds.')

    # Arguments for the script
    parser.add_argument('--output_data_dir', type=str, required=True, help='Directory where the results are saved.')
    parser.add_argument('--R_value', type=float, required=True, help='R value used for plotting the total error.')
    parser.add_argument('--local_epochs', type=int, required=True, help='local epochs value used for plotting the total error.')
    parser.add_argument('--methods', type=str, nargs='+', required=True, help='List of methods to visualize (e.g., FedAvg, LocalTrain, FedProx).')
    parser.add_argument('--mu_values', type=float, nargs='*', help='List of mu values for FedProx.')

    args = parser.parse_args()

    # Load ATR and ITR results for each method and R
    atr_dict = {}
    itr_dict = {}

    # Load results for FedAvg and LocalTrain
    for method in args.methods:
        if method != 'FedProx':
            atr, itr = load_total_results(method, args.R_value, output_data_dir=args.output_data_dir)
            atr_dict[method] = atr
            itr_dict[method] = itr

    # Load results for FedProx with different mu values
    if 'FedProx' in args.methods:
        for mu in args.mu_values:
            atr, itr = load_total_results('FedProx', args.R_value, mu=mu, output_data_dir=args.output_data_dir)
            atr_dict[f'FedProx_mu{mu}'] = atr
            itr_dict[f'FedProx_mu{mu}'] = itr
    
    # Plot and save the results
    plot_total_error(atr_dict, itr_dict, args.output_data_dir, args.R_value, args.local_epochs)

if __name__ == "__main__":
    main()