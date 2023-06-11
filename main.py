import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

import mnist 
from trainer_agent import TrainerAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

def mnist_run(device, num_filters, kernel_size, hidden_sizes, lin_dropout, conv_dropout, learning_rate, epochs, batch_size):
    # Create network layers
    network_layers = mnist.create_network_layers(1, len(num_filters), num_filters, kernel_size, hidden_sizes, lin_dropout, conv_dropout)

    # Create and train the classifier
    classifier = mnist.MNISTClassifier(device, network_layers, trainloader, testloader, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, optimizer=optim.SGD)
    train_history = classifier.train()

    # Evaluate the classifier
    eval_history = classifier.evaluate()

    # Combine train and evaluation history
    history = train_history + [eval_history]

    # Get the model structure
    model_structure = str(classifier.net)

    return history, model_structure

import re

def parse_model_output(output: str):
    action_pattern = r"Action:\s*(\w+)"
    response_pattern = r"Response:\s*(.+)"

    action_match = re.search(action_pattern, output)
    response_match = re.search(response_pattern, output)

    if action_match and response_match:
        action = action_match.group(1)
        response_content = response_match.group(1).strip()

        return {
            "action": action,
            "response_content": response_content
        }
    else:
        return None


current_hyperparams = {
    "num_filters": [32,64],
    "kernel_size": 3,
    "hidden_sizes": [128,64],
    "learning_rate": 0.001,
    "epochs": 1,
    "batch_size": 100,
    "lin_dropout": 0.5,
    "conv_dropout": 0.25,
}    

agent = TrainerAgent()
history = []
def handle_think_action(history, current_hyperparams, thought_message=None):
    model_output = agent.generate_training_action(history, current_hyperparams, thought_message=thought_message)
    print("\n****")
    print(model_output)
    print("****")

    parsed_output = parse_model_output(model_output)

    if parsed_output:
        action = parsed_output["action"]
        response_content = parsed_output["response_content"]

        if action == "Think":
            return handle_think_action(history, current_hyperparams, thought_message=response_content)
        else:
            return action, response_content
    else:
        print("Failed to parse the model output.")
        return None, None


for i in range(10):
    print(f"\n\n################PASS {i}################")
    history.append(mnist_run(device, **current_hyperparams))
    print("\n****")
    model_output = agent.generate_training_action(history, current_hyperparams)
    print(model_output)
    print("****")

    parsed_output = parse_model_output(model_output)

    if parsed_output:
        action = parsed_output["action"]
        response_content = parsed_output["response_content"]

        if action == "Change":
            new_hyperparams = eval(response_content.strip("hyperparams ="))
            current_hyperparams.update(new_hyperparams)
            print("Updated hyperparameters:", current_hyperparams)

        elif action == "Rerun":
            print("Rerunning the model with the same hyperparameters.")

        elif action == "Think":
            action, response_content = handle_think_action(history, current_hyperparams, thought_message=response_content)
            print(action)
            if action == "Change":
                new_hyperparams = eval(response_content.strip("hyperparams ="))
                current_hyperparams.update(new_hyperparams)
                print("Updated hyperparameters:", current_hyperparams)

            elif action == "Rerun":
                print("Rerunning the model with the same hyperparameters.")
    else:
        print("Failed to parse the model output.")