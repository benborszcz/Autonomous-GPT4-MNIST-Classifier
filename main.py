# Import the required libraries and functions
import torch
from trainer import mnist_training_loop

# Set the model and the number of passes for the training loop
MODEL = "gpt-4"
PASSES = 10

# Define the initial hyperparameters for the training loop
current_hyperparams = {
    "num_filters": [32, 64],
    "kernel_size": 3,
    "hidden_sizes": [128, 64],
    "learning_rate": 0.01,
    "epochs": 4,
    "batch_size": 100,
    "lin_dropout": 0.5,
    "conv_dropout": 0.25,
}

# Set the device to use for training (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Call the mnist_training_loop function with the device, model, hyperparameters, and number of passes
mnist_training_loop(device, current_hyperparams, MODEL, PASSES)