# Import dependencies
import os
import shutil
import pipeline_utils 
from random import sample

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, transforms

# Define parameters
path_to_dataset = "../../../../../../work3/s226536/datasets/oxford-iiit-pet"
number_of_samples = 5
breed = "basset_hound"

# Download dataset from open datasets in case  it is not already downloaded
datasets.OxfordIIITPet(root="../../../../../../work3/s226536/datasets", download=True)

# Get samples by breed, class, specie and breed id from txt file
samples_by_breed, class_by_id, species_by_id, breed_by_id = pipeline_utils.get_breeds(f'{path_to_dataset}/annotations/trainval.txt')

# Select a random set of samples
samples = sample(samples_by_breed[breed], number_of_samples)

# Define destination for subject-driven generation algorithm
dst = f'/zhome/d1/6/191852/MSc-thesis/experiments/03-oxford-iiit-pet/dataset'

# Delete all files in dst
if len(os.listdir(dst)) > 0:
    for filename in os.listdir(dst):
        file_path = os.path.join(dst, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# Copy selected samples to dst in order to use them in the subject-driven generation algorithm
for sample in samples:
    src = f'{path_to_dataset}/images/{sample}.jpg'
    shutil.copy(src, dst)




'''batch_size = 64

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Download training data from open datasets.
training_data = datasets.OxfordIIITPet(root="../../../../../../work3/s226536/datasets", download=True, transform=transform)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)

# Get len of dataset
print(f"Dataset training size: {training_data.__len__()}")'''


'''with torch.no_grad(): # Speed up inference
    for data in train_dataloader:
        images, labels = data
        print(labels)'''

'''
image, label = training_data.__getitem__(504)
print(label)
save_image(image, 'img1.png')
'''