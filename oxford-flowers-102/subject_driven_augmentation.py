# This file contains the code for the subject-driven fine tunning of the SD model.

# Import dependencies
import os
import time
import json
import torch
import shutil 
import random


def delete_files(dst, verbose=False):
    """
    This function deletes all files in a directory.
    :param dst: The directory to delete all files from.
    :param verbose: Indicates whether to print the information about the deleted files.
    """
    if len(os.listdir(dst)) > 0:
        for filename in os.listdir(dst):
            file_path = os.path.join(dst, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    shutil.rmtree(file_path)
            except Exception as e:
                if verbose:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == "__main__":

    # Read parameters from config file
    with open('config.json') as f: data = json.load(f)
    number_of_samples = data["number_of_samples"]
    subject_driven_technique = data["subject_driven_technique"]
    path_to_dataset = data["path_to_dataset"]


    # Create folder for saved models
    saved_models_path = f"../../../../../../work3/s226536/saved_models/flowers/{subject_driven_technique}-{number_of_samples}"
    os.makedirs(saved_models_path, exist_ok=True)

    flowers_to_generate = [str(i) for i in range(1, 103)]

    # Data augmentation generation for the selected flowers
    for flower in flowers_to_generate:

        print(f"-------------------------------------\nStarted subject-driven fine tunning for flower {flower}...\n")
        start_time = time.time()
        
        # Define destination for subject-driven generation algorithm
        dst = f'/zhome/d1/6/191852/MSc-thesis/experiments/03-oxford-iiit-pet/dataset'

        # Delete all files in dst
        delete_files(dst)

        # Copy selected random samples to dst to use them in the subject-driven generation. Samples already shuffled by prepare_dataset.py 
        path_to_flower = f'{path_to_dataset}/train/{flower}'
        for sample_image in random.sample(os.listdir(path_to_flower), number_of_samples):
            src = f'{path_to_flower}/{sample_image}'
            shutil.copy(src, dst)

        print(f"Executing subject-driven technique...\n")

        # Call subject-driven generation algorithm
        if subject_driven_technique == "textual-inversion": 
            os.system('accelerate launch /zhome/d1/6/191852/MSc-thesis/textual-inversion/huggingFace/textual_inversion.py \
                --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
                --train_data_dir="/zhome/d1/6/191852/MSc-thesis/experiments/03-oxford-iiit-pet/dataset" \
                --learnable_property="object" \
                --placeholder_token="<funny-ret>" --initializer_token="flower" \
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
        
        # Move saved_model folder to scratch space and create new saved_model folder
        shutil.move("/zhome/d1/6/191852/saved_model", saved_models_path + f"/{flower}")
        os.mkdir("/zhome/d1/6/191852/saved_model")

        # Time elapsed
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds\n")

        # Clear memory
        torch.cuda.empty_cache()