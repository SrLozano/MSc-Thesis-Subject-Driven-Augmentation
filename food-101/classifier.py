# This file contains the code to train and evaluate a classifier for the food-101 dataset

# Import dependencies
import os
import time
import json
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, transforms, AutoAugmentPolicy


class foodsModel(nn.Module):
    """
    This class defines the model. In this case, ResNet34 - Feauture extraction is used
    """

    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        # Use a pretrained model
        self.network = models.resnet34(pretrained=pretrained)

        # Freeze all the layers except the last one
        if freeze_layers == True:
            for param in self.network.parameters():
                param.requires_grad = False

        # Replace last layer. Parameters of newly constructed modules have requires_grad=True by default
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)


class EarlyStopper:
    """
    This class implements early stopping
    """
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.max_validation_accuracy = 0

    def early_stop(self, validation_accuracy):
        # Check if validation loss has decreased
        if validation_accuracy > self.max_validation_accuracy:
            self.max_validation_accuracy = validation_accuracy
            self.counter = 0
        elif validation_accuracy <= self.max_validation_accuracy:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(dataloader, model, loss_fn, optimizer):
    """
    This function trains a model with the given parameters and dataset
    :param dataloader: Training dataloader
    :param model: Model to train
    :param loss_fn: Loss function
    :param optimizer: Optimizer
    :return: Training loss
    """

    # Set model to training mode
    size = len(dataloader.dataset)
    model.train()

    # Loop over batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Training loss: {loss:>7f}  [{current}/{size}]")
    
    # Compute training accuracy
    correct = 0
    total = 0
    with torch.no_grad(): # Speed up computations. No gradients needed.
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Training accuracy: %.2f %%' % (100 * correct / total))

    return loss.item(), correct/total


def validate(dataloader, model, loss_fn):
    """
    This function evaluates a model on the validation set
    :param dataloader: Validation dataloader
    :param model: Model to validate
    :param loss_fn: Loss function
    :return: Test loss and accuracy
    """

    # Set model to evaluation mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    # Compute loss and accuracy
    validation_loss, correct = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            validation_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
 
    validation_loss /= num_batches
    correct /= size
    print(f"Validation loss: {validation_loss:>8f}")
    print(f"Validation accuracy: {(100*correct):>0.2f}% \n")

    return validation_loss, correct    


def create_plots(training_loss, validation_loss, training_accuracy, validation_accuracy, epochs):
    """
    This functions creates the plots accuracy and loss evolution in training and validation
    :param training_loss: Record of training loss values
    :param validation_loss: Record of validation loss values
    :param training_accuracy: Record of training accuracy values
    :param validation_accuracy: Record of validation accuracy values
    :param epochs: Number of epochs
    :return: It saves the accuracy and loss plots
    """
    
    # Accuracy plot
    plt.plot(training_accuracy)
    plt.plot(validation_accuracy)
    plt.title(f'Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xticks(np.arange(epochs, step=5))
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'accuracy.pdf')
    plt.close()

    # Loss plot
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.title(f'Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(np.arange(epochs, step=5))
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'loss.pdf')
    plt.close()


def create_confusion_matrix(dataloader, model):
    """
    This function evaluates a model. It shows classification report and confusion matrix.
    :param model: Model to evaluate
    :param dataloader: Evaluation dataloader
    """

    # Set model to evaluation mode
    model.eval()

    # Get predictions and ground truth
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y_true += y.cpu().detach().numpy().tolist()
            y_pred += pred.argmax(1).cpu().detach().numpy().tolist()
    
    # Show statistics
    classes_file_path = f"{path_to_dataset}/classes.txt"
    with open(classes_file_path, "r") as file:
        classes = file.readlines()
    target_names = [i.strip() for i in classes]
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(f'Test accuracy: {accuracy_score(y_true, y_pred)}')

    # Create and plot confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(15, 15))
    heatmap = sns.heatmap(cf_matrix, annot=False, cmap='Blues', cbar=True, square=False,
                          xticklabels=target_names, yticklabels=target_names)
    fig = heatmap.get_figure()
    fig.savefig('confusion_matrix.pdf')
    plt.close() 


if __name__ == "__main__":
    
    # Read parameters from config file
    with open('config.json') as f: data = json.load(f)
    
    # Define hyperparameters
    epochs = data["epochs"]
    batch_size = data["batch_size"]
    learning_rate = data["learning_rate"]
    data_augmentation = data["data_augmentation"] # auto - rand - custom - null
    path_to_dataset = data["path_to_dataset"]
    DATA_DIR = data["DATA_DIR"]

    verbose = False
    num_classes = 101
    freeze_layers = True
    

    # Time the execution
    start_time = time.time()

    # Define data augmentation transforms
    data_augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Define autoaugment transforms
    auto_augment_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        torchvision.transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor()
    ])

    # Define randaugment transforms
    rand_augment_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        torchvision.transforms.RandAugment(),
        transforms.ToTensor()
    ])

    # Define normal transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load training and validation data
    if data_augmentation == "custom":
        training_data = datasets.ImageFolder(root=f"{path_to_dataset}/train", transform=data_augmentation_transform)
        print("Using custom data augmentation")
    elif data_augmentation == "auto":
        training_data = datasets.ImageFolder(root=f"{path_to_dataset}/train", transform=auto_augment_transform)
        print("Using autoaugment for data augmentation")
    elif data_augmentation == "rand":
        training_data = datasets.ImageFolder(root=f"{path_to_dataset}/train", transform=rand_augment_transform)
        print("Using randaugment for data augmentation")
    else:
        training_data = datasets.ImageFolder(root=f"{path_to_dataset}/train", transform=transform)
        print("Not using data augmentation")

    validation_data = datasets.ImageFolder(root=f"{path_to_dataset}/valid", transform=transform)

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    # Initialize the model for this run
    model = foodsModel(num_classes).to(device)

    # Print model if verbose
    if verbose: print(model)

    # Define loss function, optimizer and early stopper
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopper(patience=5)

    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []

    # Training loop of the model
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        aux_train_loss, aux_train_accuracy = train(train_dataloader, model, loss_fn, optimizer)
        aux_validation_loss, aux_test_acurracy = validate(validation_dataloader, model, loss_fn)

        training_loss.append(aux_train_loss)
        training_accuracy.append(aux_train_accuracy)
        validation_loss.append(aux_validation_loss)
        validation_accuracy.append(aux_test_acurracy)

        # Check if early stop
        if early_stopper.early_stop(aux_test_acurracy): 
            print("Early stopping")            
            break

    # Time elapsed
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done!\n")
    print(f"Elapsed time: {elapsed_time} seconds\n")
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"Current time: {current_time}")

    # Plot the training loss and accuracy
    create_plots(training_loss, validation_loss, training_accuracy, validation_accuracy, t+1)
    
    # Get test data
    test_data = datasets.ImageFolder(root=f"{path_to_dataset}/test", transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Create confusion matrix
    create_confusion_matrix(test_dataloader, model)