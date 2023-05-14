# This script downloads the Oxford-Flowers-102 dataset and prepares it for being used in the experiments.

# Import dependencies
import os
import json
import random
import tarfile
import requests


if __name__ == "__main__":
        
    # Read parameters from config file
    with open('config.json') as f: data = json.load(f)
    training_images_percentage = data["training_images_percentage"]
    DATA_DIR = data["DATA_DIR"]

    # Define the URL, download location and extraction directory
    url = "https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz" 
    download_location = f"{DATA_DIR}/flowers-102" 

    # Remove the dataset if it already exists
    if os.path.exists(download_location):
        os.system(f'rm -rf {DATA_DIR}/flowers-102')

    # Ensure the download location and extraction directory exist
    os.makedirs(download_location, exist_ok=True)
    os.makedirs(download_location, exist_ok=True)

    # Determine the output file path
    filename = os.path.basename(url)
    output_file = os.path.join(download_location, filename)

    # Download the file
    response = requests.get(url)
    with open(output_file, "wb") as f:
        f.write(response.content)

    # Extract the contents of the tar.gz file
    with tarfile.open(output_file, "r:gz") as tar:
        tar.extractall(download_location)

    # Create training subset if necessary
    if training_images_percentage != 1:

        for folder in os.listdir(f"{download_location}/train"):

            # Get a list of all files in the folder
            file_list = os.listdir(f"{download_location}/train/{folder}")
            
            # Calculate the number of files to remove
            num_files_to_remove = int(len(file_list) * (1 - training_images_percentage))

            # Randomly select files to remove
            files_to_remove = random.sample(file_list, num_files_to_remove)

            # Remove the selected files
            for file_name in files_to_remove:
                file_path = os.path.join(f"{download_location}/train/{folder}", file_name)
                os.remove(file_path)