import os
import time
import shutil
import pipeline_utils 
from random import sample

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, transforms

# Move generated images to the corresponding folder and create annotations
generated_images_path = "/zhome/d1/6/191852/MSc-thesis/data/generated_images"

# Delete all files in generated_images_path
pipeline_utils.delete_files(generated_images_path)