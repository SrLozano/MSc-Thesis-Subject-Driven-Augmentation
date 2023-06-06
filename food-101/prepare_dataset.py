# This script  prepares the Food-101 dataset for being used in the experiments.

# Import dependencies
import os
import json
import random
import shutil
import tarfile


def reorganise_files(mode, mode_files, classes):
    """
    This function reorganises the files in the dataset.
    :param mode: The mode to reorganise (train, val or test)
    :param mode_files: The list of files in the mode
    :param classes: The list of classes in the dataset
    """
    
    # Create the new directory structure
    os.makedirs(f"{DATA_DIR}/food-101/{mode}", exist_ok=True)
    
    # Create the class directories
    for food_class in classes:
        os.makedirs(f"{DATA_DIR}/food-101/{mode}/{food_class.strip()}", exist_ok=True)

    # Move the files to the new location
    for mode_file in mode_files:
        
        # Get the class and file id
        food_class = mode_file.split("/")[0]
        file_id = mode_file.split("/")[1].strip()
        
        # Copy the file to the new location
        src = f'{DATA_DIR}/food-101/food-101/images/{food_class}/{file_id}.jpg'
        dst = f'{DATA_DIR}/food-101/{mode}/{food_class}/{file_id}.jpg'
        shutil.move(src, dst)


if __name__ == "__main__":
        
    # Read parameters from config file
    with open('config.json') as f: data = json.load(f)
    training_images_percentage = data["training_images_percentage"]
    DATA_DIR = data["DATA_DIR"]

    # Define the URL, download location and extraction directory
    output_file = f"{DATA_DIR}/food-101/food-101.tar.gz" 
    extract_location = f"{DATA_DIR}/food-101" 

    # Remove the dataset if it already exists
    if os.path.exists(extract_location):
        os.system(f'rm -rf {DATA_DIR}/food-101/food-101')  
    if os.path.exists(f"{DATA_DIR}/food-101/train"):
        os.system(f'rm -rf {DATA_DIR}/food-101/train')
    if os.path.exists(f"{DATA_DIR}/food-101/valid"):
        os.system(f'rm -rf {DATA_DIR}/food-101/valid')
    if os.path.exists(f"{DATA_DIR}/food-101/test"):
        os.system(f'rm -rf {DATA_DIR}/food-101/test')
    if os.path.exists(f"{DATA_DIR}/food-101/classes.txt"):
        os.system(f'rm -rf {DATA_DIR}/food-101/classes.txt')

    # Extract the contents of the tar.gz file
    with tarfile.open(output_file, "r:gz") as tar:
        tar.extractall(extract_location)

    # Get the list of files in the test set
    test_file_path = f"{DATA_DIR}/food-101/food-101/meta/test.txt"
    with open(test_file_path, "r") as file:
        test_lines = file.readlines()

    # Get the list of files in the train set
    train_file_path = f"{DATA_DIR}/food-101/food-101/meta/train.txt"
    with open(train_file_path, "r") as file:
        lines = file.readlines()

    # Shuffle the list of training files
    random.shuffle(lines)

    # Split the training set into a training and validation set
    train_lines = lines[:int(len(lines)*0.8)]
    val_lines = lines[int(len(lines)*0.8):]

    print(len(train_lines))
    print(len(val_lines))
    print(len(test_lines))

    # Get the list of classes
    classes_file_path = f"{DATA_DIR}/food-101/food-101/meta/classes.txt"
    with open(classes_file_path, "r") as file:
        classes = file.readlines()

    # Copy the class file to the new location
    dst = f'{DATA_DIR}/food-101/classes.txt'
    shutil.move(classes_file_path, dst)

    # Reorganise the files into train, val and test folders
    reorganise_files("train", train_lines, classes)
    reorganise_files("valid", val_lines, classes)
    reorganise_files("test", test_lines, classes)

    # Create training subset if necessary
    if training_images_percentage != 1:

        for folder in os.listdir(f"{extract_location}/train"):

            # Get a list of all files in the folder
            file_list = os.listdir(f"{extract_location}/train/{folder}")
            
            # Calculate the number of files to remove
            num_files_to_remove = int(len(file_list) * (1 - training_images_percentage))

            # Randomly select files to remove
            files_to_remove = random.sample(file_list, num_files_to_remove)

            # Remove the selected files
            for file_name in files_to_remove:
                file_path = os.path.join(f"{extract_location}/train/{folder}", file_name)
                os.remove(file_path)

    # Remove the extracted folder
    os.system(f'rm -rf {DATA_DIR}/food-101/food-101')