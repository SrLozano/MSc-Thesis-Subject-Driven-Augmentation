# Import dependencies
import os
import time
import torch
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.transforms.functional as TF 


class DeepLabV3PetsSegmentation(nn.Module):
    """
    This class defines the model. In this case, DeepLabv3 - Feauture extraction is used
    """

    def __init__(self, num_classes=3, pretrained=True):
        """
        DeepLabv3 class with custom head. DeepLabv3 model with the ResNet101 backbone
        :param num_classes: The number of output channels in the dataset masks. Defaults to 3
        """
        super().__init__()

        self.network = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)

        # Freeze all the layers except the last one
        if freeze_layers == True:
            for param in self.network.parameters():
                param.requires_grad = False

        # Replace last layer. Parameters of newly constructed modules have requires_grad=True by default
        self.network.classifier = DeepLabHead(2048, num_classes)

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
    model.train()

    losses = []
    
    # Compute loss for each batch and update model parameters
    for index, batch in enumerate(train_dataloader):
        X, y = batch        
        X, y = X.to(device), y.to(device)
        y = torch.squeeze(y)

        # Compute model predictions
        preds = model(X)['out']  
    
        # Set the gradients of all the parameters in the neural network to zero
        optimizer.zero_grad()

        # Compute the loss based on the predictions and the actual targets
        loss = loss_fn(preds, y.long()) 

        # Compute the gradients for the parameters in the neural network
        optimizer.zero_grad()   
        loss.backward() 

        # Update the parameters of the neural network based on the computed gradients
        optimizer.step()

        losses.append(loss.item())

        # Print progress
        if index % 1000 == 0:
            size = len(dataloader.dataset)
            current = (index + 1) * len(X)
            print(f"Training loss: {loss.item():>7f} [{current}/{size}]")
        
    return np.array(losses).mean()


def validate(val_dataloader, model, loss_fn):
    """
    This function validates a model with the given loss function and dataset
    :param val_dataloader: Validation dataloader
    :param model: Model to validate
    :param loss_fn: Loss function
    :return: Validation loss
    """

    # Set model to evaluation mode
    model.eval()
    
    losses = []

    # Turn off gradient calculation during model inference
    with torch.no_grad():
        for index, batch in enumerate(val_dataloader):
            X, y = batch        
            X, y = X.to(device), y.to(device)
            y = torch.squeeze(y)

            # Compute model predictions
            preds = model(X)['out']  
        
            # Compute the loss based on the predictions and the actual targets
            try:
                losses.append(loss_fn(preds, y.long()).item())
            except:
                pass

    return np.array(losses).mean()


def test(test_dataloader, model):
    """
    This function tests a model with the given dataset.
    :param test_dataloader: Test dataloader
    :param model: Model to test
    :return: Average jaccard score
    """

    # Set model to evaluation mode
    model.eval()
    
    jaccard_scores = [] 

    # Turn off gradient calculation during model inference
    with torch.no_grad():
        for index, batch in enumerate(test_dataloader):
            X, y = batch        
            X, y = X.to(device), y.to(device)
            y = torch.squeeze(y)

            # Compute model predictions
            predictions = model(X)['out']

            # Loop through the predictions and show the image and the corresponding segmentation map
            for i in range(0, len(predictions)):

                # Convert the output to a probability map and convert to numpy array
                output_probs = torch.softmax(predictions[i], dim=0).cpu().numpy() 

                # Get the predicted class for each pixel
                predicted_classes = output_probs.argmax(axis=0)
                
                # Compute the jaccard score 'weighted' to account for class imbalance
                try:
                    jc = jaccard_score(np.array(predicted_classes).flatten().astype(int), np.array(y[i].to('cpu')).flatten().astype(int), average='weighted')
                    jaccard_scores.append(jc)
                except:
                    pass
    
    return np.array(jaccard_scores).mean()


def create_plots(training_loss, validation_loss, epochs):
    """
    This functions creates the plot for the loss evolution in training and validation
    :param training_loss: Record of training loss values
    :param validation_loss: Record of validation loss values
    :param epochs: Number of epochs
    :return: It saves the accuracy and loss plots
    """
    
    # Loss plot
    plt.subplots(figsize=(15, 15))
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.title(f'Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(np.arange(epochs, step=2))
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'loss.pdf')
    plt.close() 


def visualize_segmentation_maps(images_id, DATA_DIR):
    """
    This function takes a list of image ids and displays the image and the corresponding segmentation map.
    :param images_id: A list of image ids.
    :param DATA_DIR: The path to the dataset.
    """

    # Open images
    images = []
    for image_id in images_id:
        images.append(Image.open(f"{DATA_DIR}/oxford-iiit-pet/images/{image_id}.jpg"))
        images.append(Image.open(f"{DATA_DIR}/oxford-iiit-pet/annotations/trimaps/{image_id}.png").convert("L"))

    # Calculate the number of rows and columns for the subplots
    num_rows = len(images) // 2
    num_cols = 2

    # Create a figure with n/2 rows and 2 columns of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5*num_rows))

    # Loop through the image paths and display each image in a separate subplot
    for i in range(len(images)):
        # Calculate the row and column indices for the current subplot
        row_index = i // 2
        col_index = i % 2
        
        # Display the image in the corresponding subplot
        axes[row_index, col_index].imshow(images[i])
        
        if col_index == 0:
            axes[row_index, col_index].set_title("Original image")
        else:
            axes[row_index, col_index].set_title("Segmentation mask")

    # Save figure
    plt.savefig(f"segmentation_training_data.pdf")
    plt.close() 

def get_predicted_segmentations_maps(model, images_id, DATA_DIR, transform):
    """
    This function takes a list of image ids and displays the image and the corresponding predicted segmentation map and ground truth.
    :param model: The model to use for prediction.
    :param images_id: A list of image ids.
    :param DATA_DIR: The path to the dataset.
    :param transform: The transform to apply to the images.
    """

    original_images = []
    transfored_original_images = []
    ground_truth_segmentations_maps = []
    predicted_segmentations_maps = []

    # Open original and ground truth images
    for image_id in images_id:
        original_image = Image.open(f"{DATA_DIR}/oxford-iiit-pet/images/{image_id}.jpg")
        segmentation_map = Image.open(f"{DATA_DIR}/oxford-iiit-pet/annotations/trimaps/{image_id}.png").convert("L")

        # Apply transform to the original image
        transfored_original_images.append(transform(original_image))

        original_images.append(original_image)
        ground_truth_segmentations_maps.append(segmentation_map)

    # Set model to evaluation mode
    model.eval()

    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Turn off gradient calculation during model inference
    with torch.no_grad():
        # Prepare images for model prediction
        x = np.stack(transfored_original_images)
        x = torch.from_numpy(x)
        x = x.to(device)

        # Make predictions
        predictions = model(x)['out']

        # Loop through the predictions and show the image and the corresponding segmentation map
        for i in range(0, len(predictions)):

            # Convert the output to a probability map and convert to numpy array
            output_probs = torch.softmax(predictions[i], dim=0).cpu().numpy() 

            # Get the predicted class for each pixel
            predicted_classes = output_probs.argmax(axis=0)

            # Convert the predicted classes to an image
            predicted_segmentations_maps.append(Image.fromarray(np.uint8(predicted_classes)).convert("L"))


        # Plot the images and their corresponding predicted and ground truth segmentation maps
        images_to_plot = []
        for i in range(0, len(original_images)):
            images_to_plot.append(original_images[i].resize((224, 224)))
            images_to_plot.append(ground_truth_segmentations_maps[i].resize((224, 224)))
            images_to_plot.append(predicted_segmentations_maps[i])

        # Calculate the number of rows and columns for the subplots
        num_rows = len(images_to_plot) // 3
        num_cols = 3

        # Create a figure with n/3 rows and 3 columns of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5*num_rows))

        # Loop through the image paths and display each image in a separate subplot
        for i in range(len(images_to_plot)):
            # Calculate the row and column indices for the current subplot
            row_index = i // 3
            col_index = i % 3
            
            # Display the image in the corresponding subplot
            axes[row_index, col_index].imshow(images_to_plot[i])
            
            if col_index == 0:
                axes[row_index, col_index].set_title("Original image")
            elif col_index == 1:
                axes[row_index, col_index].set_title("Segmentation mask")
            else:
                axes[row_index, col_index].set_title("Predicted segmentation mask")

        # Save figure
        current_time = datetime.now().strftime("%H:%M:%S")
        plt.savefig(f"predicted_segmentation_maps_{current_time}.pdf")
        plt.close() 


if __name__ == "__main__":

    # Hyperparameters definition
    #DATA_DIR = "/zhome/d1/6/191852"
    DATA_DIR = "../../../../../../work3/s226536/datasets"
    batch_size = 4
    verbose = False
    epochs = 10
    learning_rate = 10e-3
    freeze_layers = True

    # Clean memory. DELETE THIS
    torch.cuda.empty_cache()

    # Time the execution
    start_time = time.time()

    # Define the custom transform function to normalize the segmentation masks [0, 1, 2]
    def custom_transform(tensor):
        return (tensor * 255) - 1

    # Define normal transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(custom_transform)
    ])

    # Load training and validation data
    training_data = datasets.OxfordIIITPet(root=DATA_DIR, download=True, target_types="segmentation", transform=transform, target_transform=transform)
    validation_data = datasets.OxfordIIITPet(root=DATA_DIR, download=True, target_types="segmentation", transform=transform, target_transform=transform, split="test")

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    # Visualize segmentation maps of random samples
    images_id = random.Random(5).sample(os.listdir(f"{DATA_DIR}/oxford-iiit-pet/images") , 3)
    images_id = list(map(lambda x: x[:-4], images_id)) # Remove file extension with lambda function
    visualize_segmentation_maps(images_id, DATA_DIR)


    # Initialize the model for this run
    model = DeepLabV3PetsSegmentation(num_classes=3).to(device)

    # Print model if verbose
    if verbose: print(model)

    # Define loss function and optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    # Define early stopper
    early_stopper = EarlyStopper(patience=5)

    training_loss = []
    validation_loss = []

    # Get random images_id for visualization
    images_id = random.Random(2).sample(os.listdir(f"{DATA_DIR}/oxford-iiit-pet/images") , 3)
    images_id = list(map(lambda x: x[:-4], images_id)) # Remove file extension with lambda function


    # Training loop of the model
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        aux_training_loss = train(train_dataloader, model, loss_fn, optimizer)
        aux_validation_loss = validate(validation_dataloader, model, loss_fn)

        print(f"Training loss: {aux_training_loss:>3f} \nValidation loss: {aux_validation_loss:>3f}\n")

        # Check early stop
        if early_stopper.early_stop(aux_validation_loss): 
            print("Early stopping")            
            break

        training_loss.append(aux_training_loss)
        validation_loss.append(aux_validation_loss)

        get_predicted_segmentations_maps(model, images_id, DATA_DIR, transform)

    # Plot training and validation loss
    create_plots(training_loss, validation_loss, epochs)

    # Get test data
    os.rename(f'{DATA_DIR}/oxford-iiit-pet/annotations/test.txt', f'{DATA_DIR}/oxford-iiit-pet/annotations/validation.txt')
    os.rename(f'{DATA_DIR}/oxford-iiit-pet/annotations/final_test.txt', f'{DATA_DIR}/oxford-iiit-pet/annotations/test.txt')
    test_data = datasets.OxfordIIITPet(root=DATA_DIR, split="test", download=True, transform=transform, target_types="segmentation", target_transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Create jaccard score for the test set
    jaccard_score = test(test_dataloader, model)
    print(f"Jaccard score: {jaccard_score}")

    # Time elapsed
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done!\n")
    print(f"Elapsed time: {elapsed_time} seconds\n")
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"Current time: {current_time}")