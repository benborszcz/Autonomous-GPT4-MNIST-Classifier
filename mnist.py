import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

os.environ['OMP_NUM_THREADS'] = '1'


class CustomNet(nn.Module):
    def __init__(self, layers):
        super(CustomNet, self).__init__()
        self.layers = nn.ModuleList()
        for layer in layers:
            self.layers.append(self.create_layer(layer))

    def create_layer(self, layer):
        layer_type = layer['type']
        if layer_type == 'Conv2d':
            return nn.Conv2d(layer['in_channels'], layer['out_channels'], layer['kernel_size'], layer['stride'])
        elif layer_type == 'Linear':
            return nn.Linear(layer['in_features'], layer['out_features'])
        elif layer_type == 'Dropout':
            return nn.Dropout(layer['p'])
        elif layer_type == 'MaxPool2d':
            return nn.MaxPool2d(layer['kernel_size'], layer['stride'])
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def conv_output_shape(self, input_shape, conv_layers):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape).to(self.layers[0].weight.device)
            for layer in conv_layers:
                x = layer(x)
            return x.size()[1:]

    def forward(self, x):
        conv_layers = [layer for layer in self.layers if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d)]
        linear_start_idx = len(conv_layers)
        batch_size = x.size(0)

        for layer in self.layers[:linear_start_idx]:
            if isinstance(layer, nn.Conv2d):
                x = nn.functional.relu(layer(x))
            elif isinstance(layer, nn.MaxPool2d):
                x = layer(x)

        x = torch.flatten(x, 1)

        for layer in self.layers[linear_start_idx:]:
            if isinstance(layer, nn.Linear):
                x = nn.functional.relu(layer(x))
            elif isinstance(layer, nn.Dropout):
                x = layer(x)

        return nn.functional.log_softmax(x, dim=1)


class MNISTClassifier:
    def __init__(self, device, network_layers, trainloader, testloader, learning_rate=0.001, epochs=10, batch_size=100, optimizer=optim.SGD):
        self.device = device
        self.network_layers = network_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.trainloader = trainloader
        self.testloader = testloader

        self.net = CustomNet(self.network_layers)
        self.net.to(self.device)

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = self.optimizer(self.net.parameters(), lr=self.learning_rate, momentum=0.9)

        history = []
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / (i + 1)
            epoch_accuracy = correct / total
            history.append({"epoch": epoch + 1, "loss": epoch_loss, "accuracy": epoch_accuracy})
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")

        print("Finished Training")
        return history

    def evaluate(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Accuracy of the network on the 10000 test images: {100 * accuracy}%")
        return {"evaluation": 1, "accuracy": accuracy}


def create_network_layers(input_channels, num_conv_layers, num_filters, kernel_size, hidden_sizes, lin_dropout, conv_dropout):
    layers = []

    # Add convolutional layers
    for i in range(num_conv_layers):
        in_channels = input_channels if i == 0 else num_filters[i - 1]
        out_channels = num_filters[i]
        layers.append({'type': 'Conv2d', 'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size, 'stride': 1})

    # Add max pooling layer
    layers.append({'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2})

    # Add dropout layer
    layers.append({'type': 'Dropout', 'p': conv_dropout})

    # Calculate the input features for the first linear layer
    input_features = out_channels * ((28 - (kernel_size - 1) * num_conv_layers) // 2) ** 2

    # Add linear layers
    for i in range(len(hidden_sizes)):
        in_features = input_features if i == 0 else hidden_sizes[i - 1]
        out_features = hidden_sizes[i] if i < len(hidden_sizes) - 1 else 10
        layers.append({'type': 'Linear', 'in_features': in_features, 'out_features': out_features})

        # Add dropout layer if it's not the last linear layer
        if i < len(hidden_sizes) - 1:
            layers.append({'type': 'Dropout', 'p': lin_dropout})

    return layers



