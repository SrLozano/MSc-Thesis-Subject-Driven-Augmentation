# This file contains the code to create the pictures from the fined-tuned Stable Diffusion model.

# Import dependencies
import os
import sys
import time
import json
import torch
import shutil
import pipeline_utils 

from datetime import datetime
from diffusers import StableDiffusionPipeline

sys.path.insert(1, '/zhome/d1/6/191852/MSc-thesis') # caution: path[0] is reserved for script path (or '' in REPL)
from visualization import save_images


# Read parameters from config file
with open('config.json') as f: data = json.load(f)
images_to_generate = data["images_to_generate"] # images_to_generate should be a multiple of 5
number_of_samples = data["number_of_samples"]
subject_driven_technique = data["subject_driven_technique"]
path_to_dataset = data["path_to_dataset"]


# Get samples by breed, class, specie and breed id from txt file
samples_by_breed, class_by_id, species_by_id, breed_by_id = pipeline_utils.get_breeds(f'{path_to_dataset}/annotations/trainval.txt')

breeds_to_generate = list(samples_by_breed.keys())

# Data augmentation generation for the selected breeds
for breed in breeds_to_generate:

    # Check if model exists
    model_path = f'../../../../../../work3/s226536/saved_models/{subject_driven_technique}-{number_of_samples}/{breed}'
    if os.path.exists(model_path):
        
        print(f"-------------------------------------\nGenerating {images_to_generate} images for breed {breed}...\n")
        start_time = time.time()

        # Images generation loop
        for i in range(round(images_to_generate/5)):
            
            torch.cuda.empty_cache() # Clear memory

            placeholder_token = "<funny-ret>"

            prompts = [
                    f"A cute photo of a {placeholder_token}, golden colour, high quality, highly detailed, elegant, sharp focus"
                    ]
            keys = ["pet"]


            for i in range(len(prompts)):
                pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
                images = pipe(prompts[i], num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=5).images
                save_images(images, keys[i], "inference")

            # Rename generated images
            generated_images_path = "/zhome/d1/6/191852/MSc-thesis/data/generated_images"
            current_time = datetime.now().strftime("%H:%M:%S")
            for i, filename in enumerate(os.listdir(generated_images_path)):
                file_path = os.path.join(generated_images_path, filename)
                os.rename(file_path, generated_images_path + "/" + breed + "_" + subject_driven_technique + "_" + str(i) + "_" + str(current_time) + ".jpg")

            # Move generated images to the corresponding folder and create annotations
            dst = f'{path_to_dataset}/images'
            str_annotations = ""
            for i, filename in enumerate(os.listdir(generated_images_path)):
                src = os.path.join(generated_images_path, filename)
                # Check if file is not empty - NSFW images are empty
                if os.stat(src).st_size > 1000:
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
        print(f"Elapsed time: {elapsed_time} seconds\n")