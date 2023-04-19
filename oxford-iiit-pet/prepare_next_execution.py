# Import dependencies
import os
import json
import shutil
from datetime import datetime

print("The metadata of the previous execution is...")

# Read metadata of the previous execution
with open('config.json') as f: data = json.load(f)
print(data)

print("Preparing next execution...")

# Save results in a new folder with the current time
current_time = datetime.now().strftime("%H:%M:%S")
name_folder = f'folder_with_results_{current_time}'
os.system(f'mkdir {name_folder}')

# Copy files to the new folder
src = '/zhome/d1/6/191852/MSc-thesis/oxford-iiit-pet/accuracy.pdf'
shutil.copy(src, name_folder)
os.system('rm -rf /zhome/d1/6/191852/MSc-thesis/oxford-iiit-pet/accuracy.pdf')

src = '/zhome/d1/6/191852/MSc-thesis/oxford-iiit-pet/confusion_matrix.pdf'
shutil.copy(src, name_folder)
os.system('rm -rf /zhome/d1/6/191852/MSc-thesis/oxford-iiit-pet/confusion_matrix.pdf')

src = '/zhome/d1/6/191852/MSc-thesis/oxford-iiit-pet/loss.pdf'
shutil.copy(src, name_folder)
os.system('rm -rf /zhome/d1/6/191852/MSc-thesis/oxford-iiit-pet/loss.pdf')

# Delete dataset foldet
os.system('rm -rf /work3/s226536/datasets/oxford-iiit-pet')

print("Finished preparing next execution...")