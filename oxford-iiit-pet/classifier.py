# Import dependencies
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, transforms

class PetsModel(nn.Module):
    """
    This class defines the model. In this case, ResNet34 - Feauture extraction are used
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
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        # Check if validation loss has decreased
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
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

def test(dataloader, model, loss_fn):
    """
    This function evaluates a model on the test set
    :param dataloader: Test dataloader
    :param model: Model to evaluate
    :param loss_fn: Loss function
    :return: Test loss and accuracy
    """

    # Set model to evaluation mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    # Compute loss and accuracy
    test_loss, correct = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
 
    test_loss /= num_batches
    correct /= size
    print(f"Test loss: {test_loss:>8f}")
    print(f"Test accuracy: {(100*correct):>0.2f}% \n")

    return test_loss, correct    

def create_plots(training_loss, test_loss, training_accuracy, test_accuracy, epochs):
    """
    This functions creates the plots accuracy and loss evolution in training and validation
    :param training_loss: Record of training loss values
    :param test_loss: Record of validation loss values
    :param training_accuracy: Record of training accuracy values
    :param test_accuracy: Record of validation accuracy values
    :param epochs: Number of epochs
    :return: It saves the accuracy and loss plots
    """
    
    # Accuracy plot
    plt.plot(training_accuracy)
    plt.plot(test_accuracy)
    plt.title(f'Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xticks(np.arange(epochs, step=2))
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'accuracy.pdf')
    plt.close()

    # Loss plot
    plt.plot(training_loss)
    plt.plot(test_loss)
    plt.title(f'Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(np.arange(epochs, step=2))
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'loss.pdf')

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
    target_names = ['Abyssinian', 'American Bulldog', 'American pitbull terr', 'Basset hound', 'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau', 'English cocker spaniel', 'English setter', 'German shorthaired', 'Great pyrenees', 'Havanese', 'Japanese chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian blue', 'Saint bernard', 'Samoyed', 'Scottish terrier', 'Shiba inu', 'Siamese', 'Sphynx', 'Staffordshire bull terr', 'Wheaten terrier', 'Yorkshire terrier']
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Create and plot confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(15, 15))
    heatmap = sns.heatmap(cf_matrix, annot=False, cmap='Blues', cbar=True, square=False,
                          xticklabels=target_names, yticklabels=target_names)
    fig = heatmap.get_figure()
    fig.savefig('confusion_matrix.pdf')

if __name__ == "__main__":
    
    # Define hyperparameters
    epochs = 2
    batch_size = 16
    learning_rate = 1e-3
    data_augmentation = True
    freeze_layers = True
    verbose = False
    
    num_classes = 37
    DATA_DIR = '../../../../../../work3/s226536/datasets/oxford-iiit-pet/images'

    # Define data augmentation transforms
    data_augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])

    # Define normal transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Download training data from open datasets.
    if data_augmentation:
        training_data = datasets.OxfordIIITPet(root="../../../../../../work3/s226536/datasets", download=True, transform=data_augmentation_transform)
    else:
        training_data = datasets.OxfordIIITPet(root="../../../../../../work3/s226536/datasets", download=True, transform=transform)
    test_data = datasets.OxfordIIITPet(root="../../../../../../work3/s226536/datasets", split="test", download=True, transform=transform)

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    # Initialize the model for this run
    model = PetsModel(num_classes).to(device)

    # Print model if verbose
    if verbose: print(model)

    # Define loss function, optimizer and early stopper
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopper(patience=3)

    training_loss = []
    test_loss = []
    training_accuracy = []
    test_accuracy = []

    # Training loop of the model
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        aux_train_loss, aux_train_accuracy = train(train_dataloader, model, loss_fn, optimizer)
        aux_test_loss, aux_test_acurracy = test(test_dataloader, model, loss_fn)

        training_loss.append(aux_train_loss)
        training_accuracy.append(aux_train_accuracy)
        test_loss.append(aux_test_loss)
        test_accuracy.append(aux_test_acurracy)

        # Check if early stop
        if early_stopper.early_stop(aux_test_acurracy):             
            break

    print("Done!")

    # Plot the training loss and accuracy
    create_plots(training_loss, test_loss, training_accuracy, test_accuracy, t+1)

    # Create confusion matrix
    create_confusion_matrix(test_dataloader, model)