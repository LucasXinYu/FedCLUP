#!/bin/bash

# Start the job
echo " "
echo "MNIST experiment started at $(date)"
echo " "

cd ../src

# Function to calculate learning rates for FedProx
calculate_learning_rates() {
    mu=$1
    global_lr=$(python -c "print(1/$mu)")  # Global LR empirically set as 1/mu
    local_lr=0.01  # Empirical value
    echo $global_lr $local_lr
}

# Create methods to evaluate
fedprox_mu_values=(0.5 1.5 2.5)
methods_to_evaluate=("FedAvg" "LocalTrain")

for mu in "${fedprox_mu_values[@]}"; do
    methods_to_evaluate+=("FedProx_mu${mu}")
done

# Loop through different E (local epochs) values
for E in 5 10 15; do
    # Define shared hyperparameters with varying E
    common_hyperparameters="--E $E --r 20 --K 10 --C 1.0 --B 32 --input_dim 784 --num_classes 10 --optimizer sgd --weight_decay 0.0 --step_size 10 --gamma 1.0 --device cpu"

    # Create a subdirectory for results fx wor this value of E
    result_dir="../results/mnist/E${E}"
    mkdir -p $result_dir

    # Loop through different client classes (from 2 to 10)
    for client_classes in {2..10}; do
        start_time=$(date '+%Y-%m-%d %H:%M:%S')
        echo "${client_classes}th iteration started at $start_time"

        # Run data generation script
        python ../src/data_processing/mnist_niid.py --num_clients 300 --client_classes $client_classes --seed 1

        # Run experiments for each method
        for method in "${methods_to_evaluate[@]}"; do
            output_file="${result_dir}/output_log_${method}_${client_classes}cls.txt"
            
            if [[ "$method" == "FedAvg" || "$method" == "LocalTrain" ]]; then
                # Run FedAvg and LocalTrain with fixed learning rates
                python mnist_main.py --method "$method" --lr 0.01 $common_hyperparameters > $output_file 2>&1
            elif [[ "$method" == "FedProx_"* ]]; then
                # Extract the mu value from the method name (e.g., FedProx_mu0.5)
                mu=$(echo "$method" | sed 's/FedProx_mu//')
                # Get the learning rates for this mu value
                read global_lr local_lr < <(calculate_learning_rates $mu)
                # Run FedProx with the corresponding mu and learning rates
                python mnist_main.py --method "FedProx" --mu $mu --global_lr $global_lr --lr $local_lr $common_hyperparameters > $output_file 2>&1
            fi
        done
    done
done

# Job finished
echo " "
echo "MNIST experiment finished at $(date)"
echo " "