import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pdb
import re

# Small constant to avoid log(0)
epsilon = 1e-15

def load_total_results(local_epochs, R, mu, output_data_dir=None):
    """
    Load ATR and ITR results for a given number of local epochs, mu, and R.
    
    Args:
        local_epochs (int): Number of local epochs used during the simulation.
        R (float): The value of R used during the simulation.
        mu (float): The value of mu for FedProx.
        output_data_dir (str): Directory where ATR and ITR files are saved.
    
    Returns:
        atr (float): Full sequence of ATR.
        itr (float): Full sequence of ITR.
    """
    atr_file = os.path.join(output_data_dir, f'fedprox_mu{mu}_epochs{local_epochs}_R{R}/ATR_FedProx_mu{mu}.npy')
    itr_file = os.path.join(output_data_dir, f'fedprox_mu{mu}_epochs{local_epochs}_R{R}/ITR_FedProx_mu{mu}.npy')

    # Load the full arrays and extract the full sequence
    atr = np.load(atr_file)
    itr = np.load(itr_file)

    return atr, itr


def plot_total_error(atr_dict, itr_dict, output_dir, R_value, mu, local_epochs_list):
    """
    Plot ATR and ITR as a function of communication rounds for different local epochs.
    
    Args:
        atr_dict (dict): Dictionary with local epochs as keys and ATR values as values.
        itr_dict (dict): Dictionary with local epochs as keys and ITR values as values.
        output_dir (str): Directory where to save the plots.
    """
    # Set the figure size and marker size
    fig_size = (4, 4)
    font_size = 10
    legend_font_size = 8
    default_alpha = 1.0  # Default transparency for methods without specific alpha
    grid_alpha = 0.3  # Transparency for the grid

    # Get colors from the 'tab10' color palette
    tab10_colors = plt.get_cmap('tab10').colors

    # Plot ATR (Global Total Error) as a function of communication rounds
    plt.figure(figsize=fig_size)
    for i, local_epochs in enumerate(local_epochs_list):
        atr = atr_dict[local_epochs]
        plt.plot(range(1, len(atr)+1), atr, color=tab10_colors[i % len(tab10_colors)], 
                 linestyle='-', label=f'# Local Updates = {local_epochs}')
    
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
    plt.savefig(os.path.join(output_dir, f'Global_Total_Error_vs_Comm_Rounds_R{R_value}_mu{mu}.png'), bbox_inches='tight')
    plt.close()

    # Plot ITR (Local Total Error) as a function of communication rounds
    plt.figure(figsize=fig_size)
    for i, local_epochs in enumerate(local_epochs_list):
        itr = itr_dict[local_epochs]
        plt.plot(range(1, len(itr)+1), itr, color=tab10_colors[i % len(tab10_colors)], 
                 linestyle='-', label=f'# Local Updates = {local_epochs}')
    
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
    plt.savefig(os.path.join(output_dir, f'Local_Total_Error_vs_Comm_Rounds_R{R_value}_mu{mu}.png'), bbox_inches='tight')
    plt.close()

# Main function to handle arguments and run the visualization
def main():
    parser = argparse.ArgumentParser(description='Visualize ATR and ITR for different local epochs over Communication Rounds.')

    # Arguments for the script
    parser.add_argument('--output_data_dir', type=str, required=True, help='Directory where the results are saved.')
    parser.add_argument('--R_value', type=float, required=True, help='R value used for plotting the total error.')
    parser.add_argument('--mu', type=float, required=True, help='mu value for FedProx.')
    parser.add_argument('--local_epochs_list', type=int, nargs='+', required=True, help='List of local epochs to compare.')

    args = parser.parse_args()

    # Load ATR and ITR results for each local_epochs and R
    atr_dict = {}
    itr_dict = {}

    for local_epochs in args.local_epochs_list:
        atr, itr = load_total_results(local_epochs, args.R_value, args.mu, output_data_dir=args.output_data_dir)
        atr_dict[local_epochs] = atr
        itr_dict[local_epochs] = itr
    
    # Plot and save the results
    plot_total_error(atr_dict, itr_dict, args.output_data_dir, args.R_value, args.mu, args.local_epochs_list)

if __name__ == "__main__":
    main()