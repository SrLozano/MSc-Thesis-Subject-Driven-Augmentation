# The purpose of this script is to prepare the dataset for the segmentation pipeline.

# Import dependencies
import os
import json
import pipeline_utils 
from torchvision import datasets


if __name__ == "__main__":

    # Read parameters from config file
    with open('config.json') as f: data = json.load(f)
    training_images_percentage = data["training_images_percentage"]
    DATA_DIR = data["DATA_DIR"]


    # Download dataset from open datasets in case  it is not already downloaded
    datasets.OxfordIIITPet(root=DATA_DIR, download=True)

    # Create a subset of the training dataset
    pipeline_utils.get_training_subset(f'{DATA_DIR}/oxford-iiit-pet', training_images_percentage)

    os.rename(f'{DATA_DIR}/oxford-iiit-pet/annotations/trainval.txt', f'{DATA_DIR}/oxford-iiit-pet/annotations/original_trainval.txt')
    os.rename(f'{DATA_DIR}/oxford-iiit-pet/annotations/train_subset.txt', f'{DATA_DIR}/oxford-iiit-pet/annotations/trainval.txt')

    # Create data splits for validation and test. aka test == validation, final_test == test
    pipeline_utils.create_splits(f'{DATA_DIR}/oxford-iiit-pet')