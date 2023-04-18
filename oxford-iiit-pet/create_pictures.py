# This file contains the code to create the pictures from the fined-tuned Stable Diffusion model.

# Import dependencies
import os
import sys
import cv2
import time
import json
import torch
import random
import shutil
import numpy as np
import pipeline_utils 
import torchvision.transforms as transforms

from PIL import Image
from datetime import datetime
from diffusers.utils import load_image
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.insert(1, '/zhome/d1/6/191852/MSc-thesis') # caution: path[0] is reserved for script path (or '' in REPL)
from visualization import save_images


def generate_images(model_path, prompts, keys, subject_driven_technique, breed, samples_by_breed, path_to_dataset):
    """
    Generates images from a list of prompts using the Stable Diffusion model.
    :param model_path: path to the model
    :param prompts: list of prompts
    :param keys: list of keys
    :param subject_driven_technique: textual inversion, dreamboth, stable diffusion prompt or controlNet
    :param breed: breed of the pet
    :param samples_by_breed: dictionary with samples by breed
    :param path_to_dataset: path to the dataset
    """
    torch.cuda.empty_cache() # Clear memory

    # Generate images using SD for textual inversion, dreamboth and stable diffusion prompt
    if subject_driven_technique != "controlNet":
        for i in range(len(prompts)):
            pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
            images = pipe(prompts[i], num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=5).images
            save_images(images, keys[i], "inference")
    
    else:
        # Get a sample of images to apply controlNet to
        random_images = random.Random(41).sample(samples_by_breed[breed], 5)

        # Canny edge detection parameters
        low_threshold = 100
        high_threshold = 200
        
        # Apply controlNet to the images to get a new image that is different from the original
        for image_name in random_images:

            # Load controlNet model. It is neccesary to load it every time 
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(model_path, controlnet=controlnet, torch_dtype=torch.float16)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            
            # Apply canny edge detection to the image
            image = np.array(load_image(f"{path_to_dataset}/images/{image_name}.jpg"))
            image = cv2.Canny(image, low_threshold, high_threshold)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            canny_image = Image.fromarray(image)

            # Load the individual model components on GPU on-demand. Needed for speed purposes
            pipe.enable_model_cpu_offload()

            # Generate images using the canny edge image and controlNet
            for i in range(len(prompts)):
                print("Confirmo que estamos aqui")
                images = pipe(prompts[i], image=canny_image, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=1).images
                save_images(images, keys[i], "inference")

    
# Read parameters from config file
with open('config.json') as f: data = json.load(f)
images_to_generate = data["images_to_generate"] # images_to_generate should be a multiple of 5
number_of_samples = data["number_of_samples"] # number of samples used to be generated
subject_driven_technique = data["subject_driven_technique"] # stable-diffusion-prompt or stable-diffusion
path_to_dataset = data["path_to_dataset"] # Path to the Oxford-IIIT Pet dataset
FID_threshold = data["FID_threshold"] # FID score threshold to stop the generation process
check_quality = data["check_quality"] # Check if generated images have a good quality


# Get a sample of 37 real images from the dataset to calculate FID score as selection process
random_images = random.Random(41).sample(os.listdir(f'{path_to_dataset}/images') , 37)

# Define the FID metric with the 37 real images as reference
if check_quality == True:
    fid = FrechetInceptionDistance()
    transform = transforms.Compose([transforms.ToTensor()])

    for image_name in random_images:
        image_path = os.path.join(f'{path_to_dataset}/images', image_name)
        image_tensor = transform(Image.open(image_path)).to(torch.uint8).unsqueeze(0)
        fid.update(image_tensor, real=True)


# Get samples by breed, class, specie and breed id from txt file
samples_by_breed, class_by_id, species_by_id, breed_by_id = pipeline_utils.get_breeds(f'{path_to_dataset}/annotations/trainval.txt')
breeds_to_generate = list(samples_by_breed.keys())


# Data augmentation generation for the selected breeds
for breed in breeds_to_generate:

    # Define model path depending on the generation technique (subject-driven or not)
    if subject_driven_technique == "stable-diffusion-prompt" or subject_driven_technique == "controlNet":
        model_path = "runwayml/stable-diffusion-v1-5"
        prompts = [f"A cute photo of a {breed}, high quality, highly detailed, elegant, sharp focus"]
    else:
        model_path = f'../../../../../../work3/s226536/saved_models/{subject_driven_technique}-{number_of_samples}/{breed}'
        placeholder_token = "<funny-ret>"
        prompts = [f"A cute photo of a {placeholder_token}, high quality, highly detailed, elegant, sharp focus"]
    
    keys = ["pet"]

    # Check if model exists
    if os.path.exists(model_path) or model_path == "runwayml/stable-diffusion-v1-5":
        print(f"-------------------------------------\nGenerating {images_to_generate} images for breed {breed}...\n")
        start_time = time.time()

        # Images generation loop
        for i in range(round(images_to_generate/5)):
            generated_images_path = "/zhome/d1/6/191852/MSc-thesis/data/generated_images"
            
            if check_quality == True:
                FID = 1000 # Initialize FID score to a high value

                # Generate images until FID score is below the threshold to secure a minimum quality
                while FID > FID_threshold:
                
                    generate_images(model_path, prompts, keys, subject_driven_technique, breed, samples_by_breed, path_to_dataset)

                    # Calculate FID score for the generated images
                    for image_name in os.listdir(generated_images_path):
                        image_path = os.path.join(generated_images_path, image_name)

                        # Check if file is not empty - NSFW images are empty. Remove black images.
                        if os.stat(image_path).st_size > 1000:
                            image_tensor = transform(Image.open(image_path)).to(torch.uint8).unsqueeze(0)   
                            fid.update(image_tensor, real=False)
                        else:
                            os.remove(image_path)

                    # Get FID score
                    FID = fid.compute().item()
                    print(f"------------------ FID score of the generated images: {FID} ------------------")

                    # Score is above threshold, delete generated images and generate new ones
                    if FID > FID_threshold:
                        # Delete all files in generated_images_path
                        pipeline_utils.delete_files(generated_images_path)
                        print(f"------------------ FID score above threshold. Generating new images... ------------------")
                
                # Reset FID metric
                fid = FrechetInceptionDistance(reset_real_features=False)
            
            else:
                generate_images(model_path, prompts, keys, subject_driven_technique, breed, samples_by_breed, path_to_dataset)


            # Rename generated images
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