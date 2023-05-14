# This file contains the code to train and evaluate a classifier for the Oxford-Flowers-102 dataset

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


if __name__ == "__main__":
    
    # Read parameters from config file
    with open('config.json') as f: data = json.load(f)
    
    # Define hyperparameters
    epochs = data["epochs"]
    batch_size = data["batch_size"]
    learning_rate = data["learning_rate"]
    data_augmentation = data["data_augmentation"] # auto - rand - custom - null
    DATA_DIR = data["DATA_DIR"]

    verbose = False
    num_classes = 37
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

    # Download training data from open datasets.
    if data_augmentation == "custom":
        training_data = datasets.Flowers102(root=DATA_DIR, download=True, transform=data_augmentation_transform)
        print("Using custom data augmentation")
    elif data_augmentation == "auto":
        training_data = datasets.Flowers102(root=DATA_DIR, download=True, transform=auto_augment_transform)
        print("Using autoaugment for data augmentation")
    elif data_augmentation == "rand":
        training_data = datasets.Flowers102(root=DATA_DIR, download=True, transform=rand_augment_transform)
        print("Using randaugment for data augmentation")
    else:
        training_data = datasets.Flowers102(root=DATA_DIR, download=True, transform=transform)
        print("Not using data augmentation")
    validation_data = datasets.Flowers102(root=DATA_DIR, split="val", download=True, transform=transform)

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)


    #https://github.com/osemars/102-Flower-Classification-by-Transfer-Learning/blob/master/Image%20Classifier%20Project.py