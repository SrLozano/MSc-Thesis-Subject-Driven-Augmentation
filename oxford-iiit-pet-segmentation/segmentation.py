import os
import torch
import random
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, transforms


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


DATA_DIR = "/zhome/d1/6/191852"
batch_size = 16

# Define normal transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

training_data = datasets.OxfordIIITPet(root=DATA_DIR, download=True, transform=transform,  target_types="segmentation")
validation_data = datasets.OxfordIIITPet(root=DATA_DIR, download=True, transform=transform, target_types="segmentation", split="test")

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Visualize segmentation maps of random samples
images_id = random.Random(5).sample(os.listdir(f"{DATA_DIR}/oxford-iiit-pet/images") , 3)
images_id = list(map(lambda x: x[:-4], images_id)) # Remove file extension with lambda function
visualize_segmentation_maps(images_id, DATA_DIR)