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
import torchvision.transforms as transforms

from PIL import Image
from datetime import datetime
from diffusers.utils import load_image
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel

sys.path.insert(1, '/zhome/d1/6/191852/MSc-thesis') # caution: path[0] is reserved for script path (or '' in REPL)
from visualization import save_images


def generate_images(model_path, prompts, keys, subject_driven_technique, food_samples, food_class, path_to_dataset):
    """
    Generates images from a list of prompts using the Stable Diffusion model.
    :param model_path: path to the model
    :param prompts: list of prompts
    :param keys: list of keys
    :param subject_driven_technique: textual inversion, dreamboth, stable diffusion prompt or controlNet
    :param food_samples: list with the samples of the food that can be selected
    :param food_class: class of the food
    :param path_to_dataset: path to the dataset
    """
    torch.cuda.empty_cache() # Clear memory

    # Generate images using SD for dreamboth and stable diffusion prompt
    if subject_driven_technique == "dreambooth" or subject_driven_technique == "stable-diffusion-prompt":
        for i in range(len(prompts)):
            pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
            images = pipe(prompts[i], num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=5).images
            save_images(images, keys[i], "inference")

    # Generate images using textual inversion
    elif subject_driven_technique == "textual-inversion":
        for i in range(len(prompts)):
            pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
            pipe.load_textual_inversion(model_path)
            images = pipe(prompts[i], num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=5).images
            save_images(images, keys[i], "inference")

    # Generate images using controlNet
    elif subject_driven_technique == "controlNet":

        # Get a sample of images to apply controlNet to
        if len(food_samples) < 5:
            random_images = random.sample(food_samples, len(food_samples))
        else:
            random_images = random.sample(food_samples, 5)

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
            image = np.array(load_image(f"{path_to_dataset}/train/{food_class}/{image_name}.jpg"))
            image = cv2.Canny(image, low_threshold, high_threshold)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            canny_image = Image.fromarray(image)

            # Load the individual model components on GPU on-demand. Needed for speed purposes
            pipe.enable_model_cpu_offload()

            # Generate images using the canny edge image and controlNet
            for i in range(len(prompts)):
                images = pipe(prompts[i], image=canny_image, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=1).images
                save_images(images, keys[i], "inference")


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
    images_to_generate = data["images_to_generate"] # images_to_generate should be a multiple of 5
    number_of_samples = data["number_of_samples"] # number of samples used to be generated
    subject_driven_technique = data["subject_driven_technique"] # stable-diffusion-prompt or stable-diffusion
    path_to_dataset = data["path_to_dataset"] # Path to the Oxford-IIIT Pet dataset

    # Get the list of classes
    classes_file_path = f"{path_to_dataset}/classes.txt"
    with open(classes_file_path, "r") as file:
        classes = file.readlines()

    foods_to_generate = [i.strip() for i in classes]

    # Data augmentation generation for the selected foods
    for food in foods_to_generate:

        # Define model path depending on the generation technique (subject-driven or not)
        if subject_driven_technique == "stable-diffusion-prompt" or subject_driven_technique == "controlNet":
            model_path = "runwayml/stable-diffusion-v1-5"
            prompts = [f"A cute photo of a {food}, high quality, highly detailed, elegant, sharp focus"]
        else:
            model_path = f'../../../../../../work3/s226536/saved_models/foods/{subject_driven_technique}-{number_of_samples}/{food}'
            placeholder_token = "<funny-ret>"
            prompts = [f"A cute photo of a {placeholder_token}, high quality, highly detailed, elegant, sharp focus"]
        
        keys = ["food"]

        # Check if model exists
        if os.path.exists(model_path) or model_path == "runwayml/stable-diffusion-v1-5":
            print(f"-------------------------------------\nGenerating {images_to_generate} images for food {food}...\n")
            start_time = time.time()

            # Images generation loop
            for i in range(round(images_to_generate/5)):
                
                # Get samples of the food to generate images for in ControlNet
                food_samples = os.listdir(f"{path_to_dataset}/train/{food}")
                remove_extension = lambda file_name: file_name.replace(".jpg", "")
                food_samples = list(map(remove_extension, food_samples))
                
                # Generate images using the selected technique
                generate_images(model_path, prompts, keys, subject_driven_technique, food_samples, food, path_to_dataset)

                # Rename generated images
                current_time = datetime.now().strftime("%H:%M:%S")
                generated_images_path = "/zhome/d1/6/191852/MSc-thesis/data/generated_images"
                for i, filename in enumerate(os.listdir(generated_images_path)):
                    file_path = os.path.join(generated_images_path, filename)
                    os.rename(file_path, generated_images_path + "/" + food + "_" + subject_driven_technique + "_" + str(i) + "_" + str(current_time) + ".jpg")

                # Move generated images to the corresponding folder and create annotations
                dst = f'{path_to_dataset}/train/{food}'
                for i, filename in enumerate(os.listdir(generated_images_path)):
                    src = os.path.join(generated_images_path, filename)
                    # Check if file is not empty - NSFW images are empty
                    if os.stat(src).st_size > 1000:
                        shutil.copy(src, dst)

                # Delete all files in generated_images_path
                delete_files(generated_images_path)


            # Time elapsed
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time} seconds\n")