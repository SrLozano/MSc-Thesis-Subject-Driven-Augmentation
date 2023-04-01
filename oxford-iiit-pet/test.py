# Import dependencies
import os
import time
import shutil
import pipeline_utils 
from random import sample

# Define parameters
path_to_dataset = "../../../../../../work3/s226536/datasets/oxford-iiit-pet"
number_of_samples = 5
subject_driven_technique = "textual-inversion"
images_to_generate = 10

breed = "american_bulldog"

# Create folder for saved models
saved_models_path = f"../../../../../../work3/s226536/saved_models/{subject_driven_technique}-{number_of_samples}"
os.makedirs(saved_models_path)

# Move saved_model folder to scratch space and create new saved_model folder
shutil.move("/zhome/d1/6/191852/saved_model", saved_models_path + f"/{breed}")
os.mkdir("/zhome/d1/6/191852/saved_model")