# This script is used to prepare the next execution of the experimental pipeline.

# Import dependencies
import os
import json
import shutil
from datetime import datetime


if __name__ == "__main__":

    print("The metadata of the previous execution is...")

    # Read metadata of the previous execution
    with open('config.json') as f: data = json.load(f)
    DATA_DIR = data["DATA_DIR"]
    print(data)

    print("\nPreparing next execution...")

    # Save results in a new folder with the current time
    current_time = datetime.now().strftime("%H:%M:%S")
    name_folder = f'folder_with_results_{current_time}'
    os.system(f'mkdir {name_folder}')

    # Copy files to the new folder
    src = '/zhome/d1/6/191852/MSc-thesis/food-101/accuracy.pdf'
    shutil.copy(src, name_folder)
    os.system('rm -rf /zhome/d1/6/191852/MSc-thesis/food-101/accuracy.pdf')

    src = '/zhome/d1/6/191852/MSc-thesis/food-101/confusion_matrix.pdf'
    shutil.copy(src, name_folder)
    os.system('rm -rf /zhome/d1/6/191852/MSc-thesis/food-101/confusion_matrix.pdf')

    src = '/zhome/d1/6/191852/MSc-thesis/food-101/loss.pdf'
    shutil.copy(src, name_folder)
    os.system('rm -rf /zhome/d1/6/191852/MSc-thesis/food-101/loss.pdf')

    # Delete dataset foldet
    if os.path.exists(f"{DATA_DIR}/food-101/train"):
        os.system(f'rm -rf {DATA_DIR}/food-101/train')
    if os.path.exists(f"{DATA_DIR}/food-101/valid"):
        os.system(f'rm -rf {DATA_DIR}/food-101/valid')
    if os.path.exists(f"{DATA_DIR}/food-101/test"):
        os.system(f'rm -rf {DATA_DIR}/food-101/test')
    if os.path.exists(f"{DATA_DIR}/food-101/classes.txt"):
        os.system(f'rm -rf {DATA_DIR}/food-101/classes.txt')

    print("Finished preparing next execution...")