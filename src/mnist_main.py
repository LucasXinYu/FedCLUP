import json
from trainers.fedprox_server import FedProxServer
from trainers.args import args_parser

def load_data(train_path, test_path, evaluation_path):
    """
    Load the pre-processed training and testing data from the provided JSON files.
    """
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    with open(evaluation_path, 'r') as f:
        evaluation_data = json.load(f)
    
    return train_data, test_data, evaluation_data

def main():
    # Parse arguments from the args file
    args = args_parser()

    # Load the pre-processed data from JSON files
    train_data, test_data, evaluation_data = load_data(train_path='../data/mnist/train/all_data_uniform_train.json',
                                                test_path='../data/mnist/test/all_data_uniform_test.json',
                                                evaluation_path='../data/mnist/test/mnist_test.json')

    # Initialize the FedProx server with clients and the global model
    server = FedProxServer(args, train_data, test_data, evaluation_data)

    # Run the federated learning algorithm (FedProx) across the specified number of communication rounds
    print(f"Training the model using {args.method} with {args.model_type} model...")
    server.train(report_error="test") # For global model, we use the evaluation dataset

    # Test the final global model on all clients
    print("Running the global test across all clients...")
    server.global_test()
    server.client_test()

if __name__ == '__main__':
    main()