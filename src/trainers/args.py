import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    # Federated learning parameters
    parser.add_argument('--method', type=str, required=True, help='training method')
    parser.add_argument('--E', type=int, default=1, help='number of epochs per client (local training)')
    parser.add_argument('--r', type=int, default=40, help='number of global communication rounds')
    parser.add_argument('--K', type=int, default=10, help='number of total clients')
    parser.add_argument('--C', type=float, default=1, help='fraction of clients to sample per round')
    parser.add_argument('--B', type=int, default=50, help='local batch size for client training')

    # Model and data parameters
    parser.add_argument('--model_type', type=str, default='logistic_regression', 
                        choices=['logistic_regression', 'cnn'], 
                        help="Model to use: 'logistic_regression' or 'cnn'")
    parser.add_argument('--input_dim', type=int, default=784, help='input dimension (for MNIST it is 28x28 = 784)')
    parser.add_argument('--num_classes', type=int, default=10, help='number of output classes (for MNIST it is 10)')
    
    # FedProx specific parameters
    parser.add_argument('--mu', type=float, default=0.1, help='FedProx proximal term constant')

    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.01, help='local learning rate')
    parser.add_argument('--global_lr', type=float, default=0.01, help='global learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer: adam or sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=10, help='step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')

    # Device configuration (CPU or GPU)
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='Device to use for training and evaluation')

    # # Clients configuration
    # clients = ['Client' + str(i) for i in range(1, 11)]  # Default clients are 'Client1', 'Client2', ..., 'Client10'
    # parser.add_argument('--clients', default=clients, help='list of client names')

    args = parser.parse_args()

    return args