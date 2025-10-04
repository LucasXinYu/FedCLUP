import copy
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm
import numpy as np
import random


class Client:
    """Client class for federated learning."""

    def __init__(self, client_id, train_data, test_data, model, device='cpu', batch_size=32):
        self.client_id = client_id
        self.device = device
        self.batch_size = batch_size

        # Model initialization
        self.model = model.to(self.device)

        # Load training and testing data from JSON
        self.train_loader = self._create_dataloader(train_data)
        self.test_loader = self._create_dataloader(test_data, is_train=False)
        self.num_samples = len(train_data['x'])  # Number of training samples
        self.client_classes = list(set(train_data['y']))  # Unique classes in the training data

        # Calculate class probabilities for the naive baseline
        class_counts = np.bincount(train_data['y'], minlength=len(self.client_classes))
        self.class_probabilities = class_counts / np.sum(class_counts)
        
        self.optimizer = None  # Will be initialized later
        self.stepLR = None     # For learning rate scheduling

    def _create_dataloader(self, data, is_train=True):
        """
        Converts JSON-formatted data into PyTorch DataLoader.
        """
        x = torch.tensor(data['x'], dtype=torch.float32)
        y = torch.tensor(data['y'], dtype=torch.long)
        dataset = TensorDataset(x, y)
        shuffle = True if is_train else False
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def set_params(self, model_params):
        """
        Set the model parameters for this client.
        Args:
            model_params: The parameters to set in the model (from the global model).
        """
        self.model.load_state_dict(model_params)

    def get_params(self):
        """
        Get the current model parameters.
        Returns:
            The current state_dict of the model.
        """
        return self.model.state_dict()

    def train(self, args, global_model):
        """
        Train the client model using the FedProx method.
        Args:
            args: Training hyperparameters and configurations.
            global_model: The global model from the server.
        Returns:
            The best local model after training.
        """
        self.model.train()
        # Initialize the optimizer if not already done or is using FedAvg
        if self.optimizer is None or args.method == 'FedAvg':
            self.optimizer = self._get_optimizer(args)
            self.stepLR = StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        loss_function = nn.CrossEntropyLoss().to(self.device)        
        # Initialize the global model for FedProx and FedAvg
        if args.method == 'FedProx':
            global_model_copy = copy.deepcopy(global_model)
        
        # min_val_loss = float('inf')
        # best_model = None

        print(f'Training client {self.client_id} using {args.method} method...')
        for epoch in tqdm(range(args.E)):
            train_loss = []
            for seq, label in self.train_loader:
                seq = seq.to(self.device)
                label = label.to(self.device)
                y_pred = self.model(seq)
                self.optimizer.zero_grad()

                # Compute proximal term only for FedProx
                proximal_term = 0.0
                if args.method == 'FedProx':
                    for w, w_t in zip(self.model.parameters(), global_model_copy.parameters()):
                        proximal_term += (w - w_t).norm(2) ** 2

                # FedProx Loss: Add proximal term
                if args.method == 'FedProx':
                    loss = loss_function(y_pred, label) + (args.mu / 2) * proximal_term
                    erm_loss = loss_function(y_pred, label)
                    train_loss.append(erm_loss.item())
                else:
                    # FedAvg and LocalTrain: No proximal term
                    loss = loss_function(y_pred, label)
                    train_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

            # Update the learning rate scheduler
            self.stepLR.step()

            val_loss, _, _ = self.get_loss(self.model, data_type='test')

            print(f'Epoch {epoch + 1:03d} | Train Loss: {np.mean(train_loss):.4f} | Val Loss: {val_loss:.4f}')


        return self.model
    
    def get_loss(self, model=None, data_type='test', evaluation_data=None, class_specific=False, use_naive_baseline=False):
        """
        Evaluate the loss and accuracy on either the train, test, or evaluation set.
        
        Args:
            model: The model whose loss needs to be calculated (if None, uses self.model).
            data_type: Whether to use 'train', 'test', or 'evaluation' data for loss evaluation.
            evaluation_data: Optional, the global evaluation dataset for the 'evaluation' case.
            class_specific: Boolean, if True, evaluates only on the classes the client has seen. 
                            If False, evaluates on all classes (for global model evaluation).
            use_naive_baseline: Boolean, if True, evaluates the naive baseline model's performance.

        Returns:
            mean_loss: The mean loss over the dataset.
            correct_predictions: The number of correct predictions.
            total_samples: The number of samples used for evaluation.
        """
        if model is None and not use_naive_baseline:
            model = self.model

        model.eval()
        loss_function = nn.CrossEntropyLoss().to(self.device)
        loss_vals = []
        all_preds = []
        all_labels = []

        # Select the appropriate data loader based on data_type
        if data_type == 'train':
            data_loader = self.train_loader
        elif data_type == 'test':
            data_loader = self.test_loader
        elif data_type == 'evaluation' and evaluation_data is not None:
            data_loader = self._create_dataloader(evaluation_data, is_train=False)
        else:
            raise ValueError("Invalid data_type or missing evaluation_data")

        for seq, label in data_loader:
            with torch.no_grad():
                seq = seq.to(self.device)
                label = label.to(self.device)

                # Apply class-specific maskings
                if class_specific:
                    mask = torch.isin(label, torch.tensor(self.client_classes).to(self.device))
                    if mask.sum() == 0:
                        continue  # Skip if no valid samples for this client
                    seq, label = seq[mask], label[mask]

                if use_naive_baseline:
                    # Generate naive predictions: repeat the class probabilities for each sample
                    naive_predictions = np.tile(self.class_probabilities, (label.size(0), 1))
                    naive_predictions = torch.tensor(naive_predictions, dtype=torch.float32).to(self.device)
                    loss = loss_function(naive_predictions, label)
                    # Max prob as predicted class
                    all_preds.extend(torch.argmax(naive_predictions, dim=1).cpu().numpy())
                else:
                    y_pred = model(seq)
                    loss = loss_function(y_pred, label)
                    all_preds.extend(torch.argmax(y_pred, dim=1).cpu().numpy())

                loss_vals.append(loss.item())
                all_labels.extend(label.cpu().numpy())

        # Calculate the number of correct predictions and total samples
        num_correct = (np.array(all_preds) == np.array(all_labels)).sum()
        num_samples = len(all_labels)
        return np.mean(loss_vals), num_correct, num_samples

    def _get_optimizer(self, args):
        """Helper function to get the optimizer."""
        if args.optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            return torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)