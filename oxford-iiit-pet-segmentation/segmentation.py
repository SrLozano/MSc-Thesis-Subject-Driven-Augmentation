import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.transforms.functional as TF 


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


class UNET(nn.Module):
    
    def __init__(self, in_channels=3, classes=1):
        super(UNET, self).__init__()
        self.layers = [in_channels, 64, 128, 256, 512, 1024]
        
        self.double_conv_downs = nn.ModuleList(
            [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])
        
        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
             for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])
            
        self.double_conv_ups = nn.ModuleList(
        [self.__double_conv(layer, layer//2) for layer in self.layers[::-1][:-2]])
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

        
    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv
    
    def forward(self, x):
        # Down layers
        concat_layers = []
        
        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)
        
        concat_layers = concat_layers[::-1]
        
        # Up layers
        for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])
            
            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)
            
        x = self.final_conv(x)
        
        return x 


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
        preds = model(X)  
    
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
        if index % 100 == 0:
            size = len(dataloader.dataset)
            current = (index + 1) * len(X)
            print(f"Training loss: {loss.item():>7f} [{current}/{size}]")
        
    return np.array(losses).mean()


def validate(val_dataloader, model, loss_fn):

    # Set model to evaluation mode
    model.eval()
    
    losses = []

    # Turn off gradient calculation during model inference
    with torch.no_grad():
        for index, batch in enumerate(train_dataloader):
            X, y = batch        
            X, y = X.to(device), y.to(device)
            y = torch.squeeze(y)

            # Compute model predictions
            preds = model(X)  
        
            # Compute the loss based on the predictions and the actual targets
            losses.append(loss_fn(preds, y.long()).item())

    return np.array(losses).mean()



# Clean memory just in case. DELETE THIS
torch.cuda.empty_cache()

# Hyperparameters definition
DATA_DIR = "/zhome/d1/6/191852"
batch_size = 4
verbose = False
epochs = 10
learning_rate = 10e-3
freeze_layers = True


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
model = UNET(in_channels=3, classes=3).to(device)

# Print model if verbose
if verbose: print(model)

# Define loss function and optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
loss_fn = nn.CrossEntropyLoss(ignore_index=255)

# Training loop of the model
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    training_loss = train(train_dataloader, model, loss_fn, optimizer)
    validation_loss = validate(validation_dataloader, model, loss_fn)

    print(f"Training loss: {training_loss:>3f} \nValidation loss: {validation_loss:>3f}\n")
