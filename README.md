# Codebase for "Understanding the Accuracy-Communication Trade-off in Personalized Federated Learning"

This repository provides a PyTorch-based implementation of **FedProx-Bilevel**, which formulates the regularized PFL problem as a bilevel optimization problem and solves it using standard bilevel techniques. The repository accompanies the paper "Understanding the Accuracy-Communication Trade-off in Personalized Federated Learning" and provides code to replicate the results discussed in the paper.

## Folder Structure

Here’s the example structure of the repository:

```python
fedprox-bilevel/
├── data/                         # Directory for generated datasets (empty initially)
├── notebooks/                    # Jupyter notebooks for replicating experiments
├── src/                          # Core source code including data preprocessing, models, and trainers
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Usage

To run the experiments and replicate the results from the paper, navigate to the `notebooks` folder and use the corresponding Jupyter notebooks. The notebooks will guide you through generating datasets, training the models using the FedProx algorithm, and evaluating the results.

## Dependencies

This project uses Python with PyTorch as the primary framework. You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.
