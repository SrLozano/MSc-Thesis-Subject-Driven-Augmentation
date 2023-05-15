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

sys.path.insert(1, '/zhome/d1/6/191852/MSc-thesis') # caution: path[0] is reserved for script path (or '' in REPL)
from visualization import save_images


def generate_images(model_path, prompts, keys, subject_driven_technique, flower_samples, flower_class, path_to_dataset):
    """
    Generates images from a list of prompts using the Stable Diffusion model.
    :param model_path: path to the model
    :param prompts: list of prompts
    :param keys: list of keys
    :param subject_driven_technique: textual inversion, dreamboth, stable diffusion prompt or controlNet
    :param flower_samples: list with the samples of the flower that can be selected
    :param flower_class: class of the flower
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
        if len(flower_samples) < 5:
            random_images = random.sample(flower_samples, len(flower_samples))
        else:
            random_images = random.sample(flower_samples, 5)

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
            image = np.array(load_image(f"{path_to_dataset}/train/{flower_class}/{image_name}.jpg"))
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


if __name__ == "__main__":
    
    # Read parameters from config file
    with open('config.json') as f: data = json.load(f)
    images_to_generate = data["images_to_generate"] # images_to_generate should be a multiple of 5
    number_of_samples = data["number_of_samples"] # number of samples used to be generated
    subject_driven_technique = data["subject_driven_technique"] # stable-diffusion-prompt or stable-diffusion
    path_to_dataset = data["path_to_dataset"] # Path to the Oxford-IIIT Pet dataset

    flowers_to_generate = [str(i) for i in range(1, 103)]

    # Data augmentation generation for the selected flowers
    for flower in flowers_to_generate:

        # Define model path depending on the generation technique (subject-driven or not)
        if subject_driven_technique == "stable-diffusion-prompt" or subject_driven_technique == "controlNet":
            model_path = "runwayml/stable-diffusion-v1-5"
            prompts = [f"A cute photo of a {flower}, high quality, highly detailed, elegant, sharp focus"]
        else:
            model_path = f'../../../../../../work3/s226536/saved_models/flowers/{subject_driven_technique}-{number_of_samples}/{flower}'
            placeholder_token = "<funny-ret>"
            prompts = [f"A cute photo of a {placeholder_token}, high quality, highly detailed, elegant, sharp focus"]
        
        keys = ["flower"]

        # Check if model exists
        if os.path.exists(model_path) or model_path == "runwayml/stable-diffusion-v1-5":
            print(f"-------------------------------------\nGenerating {images_to_generate} images for flower {flower}...\n")
            start_time = time.time()

            # Images generation loop
            for i in range(round(images_to_generate/5)):
                
                # Get samples of the flower to generate images for in ControlNet
                flower_samples = os.listdir(f"{path_to_dataset}/train/{flower}")
                remove_extension = lambda file_name: file_name.replace(".jpg", "")
                flower_samples = list(map(remove_extension, flower_samples))
                
                # Generate images using the selected technique
                generate_images(model_path, prompts, keys, subject_driven_technique, flower_samples, flower, path_to_dataset)

                # Rename generated images
                current_time = datetime.now().strftime("%H:%M:%S")
                generated_images_path = "/zhome/d1/6/191852/MSc-thesis/data/generated_images"
                for i, filename in enumerate(os.listdir(generated_images_path)):
                    file_path = os.path.join(generated_images_path, filename)
                    os.rename(file_path, generated_images_path + "/" + flower + "_" + subject_driven_technique + "_" + str(i) + "_" + str(current_time) + ".jpg")

                # Move generated images to the corresponding folder and create annotations
                dst = f'{path_to_dataset}/train/{flower}'
                for i, filename in enumerate(os.listdir(generated_images_path)):
                    src = os.path.join(generated_images_path, filename)
                    # Check if file is not empty - NSFW images are empty
                    if os.stat(src).st_size > 1000:
                        shutil.copy(src, dst)

                # Delete all files in generated_images_path
                pipeline_utils.delete_files(generated_images_path)


            # Time elapsed
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time} seconds\n")