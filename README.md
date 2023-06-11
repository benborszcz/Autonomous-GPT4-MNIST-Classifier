# Autonomous GPT-4 MNIST Classifier

This project demonstrates an autonomous GPT-4 MNIST classifier that trains a neural network on the MNIST dataset. The GPT-4 model is used to suggest changes to the hyperparameters of the neural network, aiming to improve its performance.

## Instructions

To use this project, follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running `pip install -r requirements.txt`.

3. Create a `.env` file in the project's root directory.

4. Add your OpenAI API key to the `.env` file in the following format:

   ```
   OPENAI_API_KEY=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
   ```

5. Run the `main.py` script to start the training process and see the GPT-4 model's suggestions for hyperparameter changes.

## Overview

The project consists of several components:

1. A custom neural network for classifying MNIST digits.
2. A training loop that trains the neural network on the MNIST dataset.
3. An agent that uses the GPT-4 model to suggest changes to the neural network's hyperparameters.
4. A main script that iteratively trains the neural network and updates its hyperparameters based on the GPT-4 model's suggestions.

## Example Output

The output shows the training progress and the GPT-4 model's suggestions for hyperparameter changes. Here's an example:

```
#### PASS 0 ####
Epoch 1, Loss: 0.44325509246438743, Accuracy: 0.8576
Epoch 2, Loss: 0.11568690035802623, Accuracy: 0.9651833333333333
Epoch 3, Loss: 0.08473589405572662, Accuracy: 0.97485
Epoch 4, Loss: 0.06580043633080399, Accuracy: 0.9801
Finished Training
Accuracy of the network on the 10000 test images: 97.88%

**** Asking GPT for Changes ****
Action: Think
Response: Based on the provided history, it seems that the model is performing well with the current hyperparameters. However, there is room for improvement. I suggest trying out different combinations of hyperparameters such as increasing the number of filters, changing the kernel size, and adjusting the learning rate. Additionally, experimenting with different dropout rates and hidden layer sizes may also improve the model's performance.

**** Asking GPT for follow up ****
Action: Change
Response: hyperparams = {"num_filters": [64, 128], "kernel_size": 5, "hidden_sizes": [256, 128], "learning_rate": 0.001, "epochs": 6, "batch_size": 100, "lin_dropout": 0.3, "conv_dropout": 0.2}

Old hyperparameters: {'num_filters': [32, 64], 'kernel_size': 3, 'hidden_sizes': [128, 64], 'learning_rate': 0.01, 'epochs': 4, 'batch_size': 100, 'lin_dropout': 0.5, 'conv_dropout': 0.25}
Updated hyperparameters: {'num_filters': [64, 128], 'kernel_size': 5, 'hidden_sizes': [256, 128], 'learning_rate': 0.001, 'epochs': 6, 'batch_size': 100, 'lin_dropout': 0.3, 'conv_dropout': 0.2}

#### PASS 1 ####
Epoch 1, Loss: 0.7624234417329232, Accuracy: 0.77315
Epoch 2, Loss: 0.22502373995880287, Accuracy: 0.9330333333333334
Epoch 3, Loss: 0.14226289555430413, Accuracy: 0.9577166666666667
Epoch 4, Loss: 0.10920122229494154, Accuracy: 0.9676666666666667
Epoch 5, Loss: 0.08761066298310956, Accuracy: 0.9736166666666667
Epoch 6, Loss: 0.07613769611654182, Accuracy: 0.9773333333333334
Finished Training
Accuracy of the network on the 10000 test images: 97.94%
```

The GPT-4 model analyzes the training history and suggests changes to the hyperparameters, which are then applied to the neural network. The process continues iteratively, aiming to improve the classifier's performance.
