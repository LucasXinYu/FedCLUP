import copy
import random
import numpy as np
import torch
from tqdm import tqdm
from models.logistic_regression import LogisticRegression  # Your logistic regression model for MNIST
from models.CNN import CNN
from trainers.fedprox_client import Client  # Updated Client class
from trainers.args import args_parser
import pdb

class FedProxServer:
    def __init__(self, args, train_data, test_data, evaluation_data=None):
        """
        Initialize the FedProxServer, set up global model and client instances.
        
        Args:
            args: Configuration and hyperparameters for FedProx training.
            train_data: Training data, a dictionary where keys are client IDs and values are data.
            test_data: Testing data, similar structure to train_data.
        """
        self.args = args
        
        # Choose the model type: logistic regression or CNN
        if args.model_type == 'logistic_regression':
            self.global_model = LogisticRegression(input_dim=args.input_dim, num_classes=args.num_classes).to(args.device)
        elif args.model_type == 'cnn':
            self.global_model = CNN(input_dim=args.input_dim, num_classes=args.num_classes).to(args.device)
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        # Create clients using the Client class
        self.clients = self.initialize_clients(train_data, test_data)
        self.global_train_losses = []  # Initialize an empty list to store global training losses
        self.client_avg_train_losses = []

        if evaluation_data is not None:
            self.evaluation_data = evaluation_data
            self.global_evaluation_losses = []
            self.global_evaluation_accuracies = []
            self.client_avg_evaluation_losses = []
            self.client_avg_evaluation_accuracies = []


    # Initialize clients with the correct train and test data
    def initialize_clients(self, train_data, test_data):
        clients = []
        
        # Get the list of all available client names
        all_client_names = train_data['users']
        
        # If K is less than the number of total clients, randomly select K clients
        if self.args.K < len(all_client_names):
            random.seed(1)
            selected_client_names = random.sample(all_client_names, self.args.K)
        else:
            selected_client_names = all_client_names  # Use all clients if K is greater than or equal to the total clients
        
        # Create Client instances for the selected clients
        for client_name in selected_client_names:
            client = Client(client_id=client_name, 
                            train_data=train_data['user_data'][client_name], 
                            test_data=test_data['user_data'][client_name], 
                            model=copy.deepcopy(self.global_model), 
                            device=self.args.device,
                            batch_size=self.args.B)
            clients.append(client)
        
        return clients

    def train(self, report_error = None):
        """
        Train the global model using FedAvg, FedProx, or LocalTrain for multiple communication rounds.
        """
        for t in tqdm(range(self.args.r)):
            print(f'Round {t + 1}:')
            # Calculate the training loss for the global model across all clients
            global_train_loss = self.get_training_loss(self.global_model, self.clients)
            self.global_train_losses.append(global_train_loss)
            avg_client_train_loss = self.get_avg_client_training_loss(self.clients)
            self.client_avg_train_losses.append(avg_client_train_loss)
            print(f"At round {t}, the global model training loss: {global_train_loss:.4f}")
            print(f"At round {t}, the average client training loss: {avg_client_train_loss:.4f}")
            if self.evaluation_data is not None and report_error is not None:
                if report_error == "test":
                    total_eval_loss, total_eval_correct, total_eval_samples = self._compute_loss(
                        data_type='test', class_specific=True
                    )
                    global_eva_loss, global_eva_correct, global_eva_samples = self.clients[0].get_loss(
                        self.global_model, data_type='evaluation', evaluation_data=self.evaluation_data) # For the global model, we always consider the evaluation loss
                elif report_error == "evaluation":
                    total_eval_loss, total_eval_correct, total_eval_samples = self._compute_loss(
                        data_type='evaluation', class_specific=True
                    )
                    global_eva_loss, global_eva_correct, global_eva_samples = self.clients[0].get_loss(
                        self.global_model, data_type='evaluation',  evaluation_data=self.evaluation_data)
                avg_eval_loss = total_eval_loss / total_eval_samples
                avg_eval_accuracy = total_eval_correct / total_eval_samples
                self.client_avg_evaluation_losses.append(avg_eval_loss)
                self.client_avg_evaluation_accuracies.append(avg_eval_accuracy)
                print(f"Average evaluation loss across clients: {avg_eval_loss:.6f}")
                print(f"Average evaluation accuracy across clients: {avg_eval_accuracy:.4f}")
        

                global_eva_accuracy = global_eva_correct / global_eva_samples
                self.global_evaluation_losses.append(global_eva_loss)
                self.global_evaluation_accuracies.append(global_eva_accuracy)
                print(f"Global Evaluation Loss: {global_eva_loss:.6f}")
                print(f"Global Evaluation Accuracy: {global_eva_accuracy:.4f}")

            # Client selection: Randomly select a subset of clients
            m = max(int(self.args.C * self.args.K), 1)
            selected_clients = random.sample(self.clients, m)

            if self.args.method == 'FedAvg':
                # FedAvg: Dispatch the global model to selected clients
                self.dispatch(selected_clients)

                # Perform local updates on the selected clients (FedAvg)
                self.client_update(selected_clients, self.args)

                # Aggregate the client models to update the global model (FedAvg)
                self.aggregation(selected_clients, self.args)

            elif self.args.method == 'FedProx':
                # FedProx: No dispatch (no resetting of local models)
                
                # Perform local updates on the selected clients (FedProx)
                self.client_update(selected_clients, self.args)

                # Aggregate the client models to update the global model (FedProx)
                self.aggregation(selected_clients, self.args)

            elif self.args.method == 'LocalTrain':
                # LocalTrain: No dispatch, clients continue local training

                # Perform local updates on the selected clients (LocalTrain)
                self.client_update(selected_clients, self.args)

                # Aggregate the client models (LocalTrain, for evaluation purpose)
                self.aggregation(selected_clients, self.args)

        return self.global_model  # Return the final global model after training

    def aggregation(self, selected_clients, args):
        """
        Aggregates the model parameters from the selected clients using either FedAvg, LocalTrain, or FedProx.

        Args:
            selected_clients: The clients that were selected for this round.
            args: Arguments containing the method type ('FedAvg', 'LocalTrain', or 'FedProx') and other hyperparameters.
        """
        total_samples = sum(client.num_samples for client in selected_clients)

        # Initialize a dictionary to hold the aggregated parameters
        aggregated_params = {k: torch.zeros_like(v.data) for k, v in self.global_model.named_parameters()}
        for client in selected_clients:
            client_weight = client.num_samples / total_samples
            for k, v in client.model.named_parameters():
                aggregated_params[k] += v.data * client_weight

        if args.method == 'FedAvg' or args.method == 'LocalTrain':
            # Update the global model with the aggregated parameters (FedAvg / LocalTrain)
            for k, v in self.global_model.named_parameters():
                v.data = aggregated_params[k].data.clone()

        elif args.method == 'FedProx':
            # Update the global model by applying the proximal term (FedProx)
            for k, v in self.global_model.named_parameters():
                # Weighted average of the previous global model and the aggregated parameters with the proximal term
                v.data =  (1 - args.mu * args.global_lr) * v.data.clone() + (args.mu * args.global_lr) * aggregated_params[k].data.clone()
                # In this version of Fedprox we assume equal weight for each of the client
        else:
            raise ValueError("the method argument cannot be identified, it should be in ['FedProx', 'FedAvg', 'LocalTrain']")

    def dispatch(self, selected_clients):
        """
        Dispatch the global model parameters to the selected clients.
        
        Args:
            selected_clients: The clients that will receive the global model parameters.
        """
        for client in selected_clients:
            client.set_params(self.global_model.state_dict())

    def client_update(self, selected_clients, args):
        """
        Updates the selected clients by running local training (FedProx style).
        
        Args:
            selected_clients: The clients that will perform local updates.
        """
        for client in selected_clients:
            # Perform local training using the client's train() method with the global model passed in
            client.train(args, self.global_model)

    def get_training_loss(self, global_model, clients):
        """
        Calculate the average training loss of the global model across all clients.

        Args:
            global_model: The global model after aggregation.
            clients: List of clients participating in the training.

        Returns:
            average_train_loss: The average training loss across all clients.
        """
        total_loss = 0
        total_samples = 0

        for client in clients:
            train_loss, _, _ = client.get_loss(global_model, data_type='train')
            num_samples = len(client.train_loader.dataset)  # Number of training samples in the client's dataset
            total_loss += train_loss * num_samples
            total_samples += num_samples

        average_train_loss = total_loss / total_samples
        return average_train_loss
    
    def get_avg_client_training_loss(self, clients):
        """
        Calculate the average training loss for each client's own model on its own dataset.

        Args:
            clients: List of clients participating in the training.

        Returns:
            avg_client_train_loss: The average training loss across all clients' own models.
        """
        total_loss = 0
        total_samples = 0

        for client in clients:
            # Get the loss of the client's own model on its own training data
            client_train_loss, _, _ = client.get_loss(client.model, data_type='train')
            num_samples = len(client.train_loader.dataset)  # Number of training samples in the client's dataset
            total_loss += client_train_loss * num_samples
            total_samples += num_samples

        avg_client_train_loss = total_loss / total_samples
        return avg_client_train_loss

    def client_test(self):
        """
        Test each client's model on its own dataset and print individual test accuracy.
        """
        # Calculate and print training loss and accuracy
        # DOUBLE CHECK HERE...
        total_train_loss, total_train_correct, total_train_samples = self._compute_loss(data_type='train')
        total_test_loss, total_test_correct, total_test_samples = self._compute_loss(data_type='test')
        
        print(f"Average training loss across clients: {total_train_loss / total_train_samples:.6f}")
        print(f"Average training accuracy across clients: {total_train_correct / total_train_samples:.4f}")
        print(f"Average test loss across clients: {total_test_loss / total_test_samples:.6f}")
        print(f"Average test accuracy across clients: {total_test_correct / total_test_samples:.4f}")

        losses_str = ','.join([f'{loss:.6f}' for loss in self.client_avg_train_losses])
        print(f"Average Client Training Losses: {losses_str}")   
        eva_losses_str = ','.join([f'{loss:.6f}' for loss in self.client_avg_evaluation_losses])
        print(f"Average Client Evaluation losses: {eva_losses_str}")   
        eva_accuracy_str = ','.join([f'{loss:.4f}' for loss in self.client_avg_evaluation_accuracies])
        print(f"Average Client Evaluation Accuracies: {eva_accuracy_str}") 

        # Evaluate model and calculate improvement over naive baseline
        if self.evaluation_data is not None:
            # Compute evaluation loss using the client's model
            total_eval_loss, total_eval_correct, total_eval_samples = self._compute_loss(
                data_type='evaluation', class_specific=True
            )

            # Compute evaluation loss using the naive baseline
            total_naive_loss, total_naive_correct, _ = self._compute_loss(
                data_type='evaluation', class_specific=True, use_naive_baseline=True
            )

            # Print evaluation results for each client and overall performance
            avg_eval_loss = total_eval_loss / total_eval_samples
            avg_eval_accuracy = total_eval_correct / total_eval_samples
            avg_naive_loss = total_naive_loss / total_eval_samples
            avg_naive_accuracy = total_naive_correct / total_eval_samples

            print(f"Average evaluation loss across clients: {avg_eval_loss:.6f}")
            print(f"Average evaluation accuracy across clients: {avg_eval_accuracy:.4f}")
            print(f"Average naive baseline loss across clients: {avg_naive_loss:.6f}")
            print(f"Average naive baseline accuracy across clients: {avg_naive_accuracy:.4f}")
            print(f"Average improvement over naive baseline: {(avg_naive_loss - avg_eval_loss) / avg_naive_loss:.4f}")

    def _compute_loss(self, data_type, use_naive_baseline=False, class_specific=False):
        """
        Helper function to compute loss and accuracy for a specific data type (train/test/evaluation).
        Allows for computation using a naive baseline.
        """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for client in self.clients:
            client_loss, client_correct, client_samples = client.get_loss(
                model=client.model,
                data_type=data_type,
                evaluation_data=self.evaluation_data,
                class_specific=class_specific,
                use_naive_baseline = use_naive_baseline
            )
            total_loss += client_loss * client_samples
            total_correct += client_correct
            total_samples += client_samples
            accuracy = client_correct / client_samples
            print(f"{data_type.capitalize()} Loss for client {client.client_id}: {client_loss:.4f}, Accuracy: {accuracy:.4f}")

        return total_loss, total_correct, total_samples

    def global_test(self):
        """
        Test the global model on all clients' datasets and compute the overall test accuracy.
        """
        self.global_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Test the global model on each client's dataset
        for client in self.clients:
            test_loss, correct, samples = client.get_loss(self.global_model, data_type='test')  # Test using the global model
            total_correct += correct
            total_samples += samples
            total_loss += test_loss * samples

        global_train_loss = self.get_training_loss(self.global_model, self.clients)
        print(f"Global Train Loss across all clients: {global_train_loss:.6f}")     
        train_losses_str = ','.join([f'{loss:.6f}' for loss in self.global_train_losses])
        print(f"Global Train losses: {train_losses_str}")   
        eva_losses_str = ','.join([f'{loss:.6f}' for loss in self.global_evaluation_losses])
        print(f"Global Evaluation losses: {eva_losses_str}")   
        eva_accuracy_str = ','.join([f'{loss:.4f}' for loss in self.global_evaluation_accuracies])
        print(f"Global Evaluation Accuracies: {eva_accuracy_str}")   
        # Compute global accuracy across all clients
        average_test_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"Global Test Loss: {average_test_loss:.6f}")
        print(f"Global Test Accuracy: {accuracy:.4f}")

        if self.evaluation_data is not None:
            eva_loss, eva_correct, eva_samples = client.get_loss(self.global_model, data_type='evaluation',  evaluation_data=self.evaluation_data)
            evaluation_accuracy = eva_correct / eva_samples
            print(f"Global Evaluation Loss: {eva_loss:.6f}")
            print(f"Global Evaluation Accuracy: {evaluation_accuracy:.4f}")
    
