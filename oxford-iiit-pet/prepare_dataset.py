# Import dependencies
import os
import pipeline_utils 
from torchvision import datasets


# Define parameters
training_images_percentage = 1
subject_driven_augmentation = False
DATA_DIR = "../../../../../../work3/s226536/datasets"


# Download dataset from open datasets in case  it is not already downloaded
datasets.OxfordIIITPet(root=DATA_DIR, download=True)

# Create a subset of the training dataset
pipeline_utils.get_training_subset(f'{DATA_DIR}/oxford-iiit-pet', training_images_percentage)

os.rename(f'{DATA_DIR}/oxford-iiit-pet/annotations/trainval.txt', f'{DATA_DIR}/oxford-iiit-pet/annotations/original_trainval.txt')
os.rename(f'{DATA_DIR}/oxford-iiit-pet/annotations/train_subset.txt', f'{DATA_DIR}/oxford-iiit-pet/annotations/trainval.txt')

# Create data splits for validation and test. aka test == validation, final_test == test
pipeline_utils.create_splits(f'{DATA_DIR}/oxford-iiit-pet')

# Run subject-driven augmentation
if subject_driven_augmentation:
    os.system('python3 /zhome/d1/6/191852/MSc-thesis/oxford-iiit-pet/subject_driven_augmentation.py')