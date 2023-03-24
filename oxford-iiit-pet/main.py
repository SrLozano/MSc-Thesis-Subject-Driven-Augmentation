# Import dependencies
import os
import time
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
subject_driven_technique = "dreambooth"

# Download dataset from open datasets in case  it is not already downloaded
datasets.OxfordIIITPet(root="../../../../../../work3/s226536/datasets", download=True)

# Get samples by breed, class, specie and breed id from txt file
samples_by_breed, class_by_id, species_by_id, breed_by_id = pipeline_utils.get_breeds(f'{path_to_dataset}/annotations/trainval.txt')

#breeds_to_generate = list(samples_by_breed.keys())
breeds_to_generate = ["basset_hound"]

# Data augmentation generation for the selected breeds
for breed in breeds_to_generate:

    print(f"-------------------------------\n\nStarted data generation process for breed {breed}...\n")
    start_time = time.time()
    
    # Select a random set of samples
    samples = sample(samples_by_breed[breed], number_of_samples)

    # Define destination for subject-driven generation algorithm
    dst = f'/zhome/d1/6/191852/MSc-thesis/experiments/03-oxford-iiit-pet/dataset'

    # Delete all files in dst
    pipeline_utils.delete_files(dst)

    # Copy selected samples to dst in order to use them in the subject-driven generation algorithm
    for sample in samples:
        src = f'{path_to_dataset}/images/{sample}.jpg'
        shutil.copy(src, dst)

    print(f"Executing subject-driven technique...\n")

    # Call subject-driven generation algorithm
    if subject_driven_technique == "textual-inversion": 
        os.system('accelerate launch /zhome/d1/6/191852/MSc-thesis/textual-inversion/huggingFace/textual_inversion.py \
            --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
            --train_data_dir="/zhome/d1/6/191852/MSc-thesis/experiments/03-oxford-iiit-pet/dataset" \
            --learnable_property="object" \
            --placeholder_token="<funny-ret>" --initializer_token="dog" \
            --resolution=512 \
            --train_batch_size=1 \
            --gradient_accumulation_steps=4 \
            --max_train_steps=3000 \
            --learning_rate=5.0e-04 --scale_lr \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --output_dir="/zhome/d1/6/191852/saved_model" \
            ') 
    elif subject_driven_technique == "dreambooth":
        os.system('accelerate launch /zhome/d1/6/191852/MSc-thesis/dreambooth/train_dreambooth.py \
            --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
            --instance_data_dir="/zhome/d1/6/191852/MSc-thesis/experiments/03-oxford-iiit-pet/dataset" \
            --output_dir="/zhome/d1/6/191852/saved_model" \
            --instance_prompt="<funny-ret>" \
            --resolution=512 \
            --train_batch_size=1 \
            --gradient_accumulation_steps=1 \
            --learning_rate=5e-6 \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --max_train_steps=400 \
            ')
    
    print(f"Generating images...\n")

    # Create pictures for the selected breed
    os.system('python3 /zhome/d1/6/191852/MSc-thesis/oxford-iiit-pet/create_pictures.py')

    # Rename generated images
    generated_images_path = "/zhome/d1/6/191852/MSc-thesis/data/generated_images"
    for i, filename in enumerate(os.listdir(generated_images_path)):
        file_path = os.path.join(generated_images_path, filename)
        os.rename(file_path, generated_images_path + "/" + breed + "_" + subject_driven_technique + "_" + str(i) + ".jpg")
    
    # Move generated images to the corresponding folder and create annotations
    generated_images_path = "/zhome/d1/6/191852/MSc-thesis/data/generated_images"
    dst = f'{path_to_dataset}/images'
    str_annotations = ""
    for i, filename in enumerate(os.listdir(generated_images_path)):
        src = os.path.join(generated_images_path, filename)
        shutil.copy(src, dst)
        str_annotations = str_annotations + "\n" + filename.split('.')[0] + f" {class_by_id[breed]}" + f" {species_by_id[breed]}" + f" {breed_by_id[breed]}" 

    # Delete all files in generated_images_path
    pipeline_utils.delete_files(generated_images_path)

    # Add annotations to the generated images
    with open(f'{path_to_dataset}/annotations/trainval.txt', 'a') as file:
        file.write(str_annotations)

    # Time elapsed
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")