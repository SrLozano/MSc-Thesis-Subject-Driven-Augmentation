# This file contains the code to create the pictures from the Stable Diffusion model using controlNet. Target task: Segmentation.

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

from PIL import Image
from datetime import datetime
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel

sys.path.insert(1, '/zhome/d1/6/191852/MSc-thesis') # caution: path[0] is reserved for script path (or '' in REPL)
from visualization import save_images


def generate_image(model_path, prompts, keys, path_to_dataset, image_name):
    """
    Generates images from a list of prompts using the Stable Diffusion model.
    :param model_path: path to the model
    :param prompts: list of prompts
    :param keys: list of keys
    :param path_to_dataset: path to the dataset
    :param image_name: name of the image taken as reference to generate the images
    :return: list of random images selected to perform the controlNet image generation
    """
    
    # Canny edge detection parameters
    low_threshold = 100
    high_threshold = 200

    # Clear memory
    torch.cuda.empty_cache() 
    
    # Load controlNet model. It is neccesary to load it every time 
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(model_path, controlnet=controlnet, torch_dtype=torch.float16)

    # Memory efficient attention
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    
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
        images = pipe(prompts[i], image=canny_image, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=1).images
        save_images(images, keys[i], "inference")


if __name__ == "__main__":

    # Read parameters from config file
    with open('config.json') as f: data = json.load(f)
    images_to_generate = data["images_to_generate"] # images_to_generate should be a multiple of 5
    path_to_dataset = data["path_to_dataset"] # Path to the Oxford-IIIT Pet dataset


    # Get samples by breed, class, specie and breed id from txt file
    samples_by_breed, class_by_id, species_by_id, breed_by_id = pipeline_utils.get_breeds(f'{path_to_dataset}/annotations/trainval.txt')
    breeds_to_generate = list(samples_by_breed.keys())


    # Data augmentation generation for the selected breeds
    for breed in breeds_to_generate[5:37]:

        # Get a sample of images to apply controlNet to
        controlNet_sample_images = random.sample(samples_by_breed[breed], images_to_generate)

        # Define model path depending for controlNet
        model_path = "runwayml/stable-diffusion-v1-5"
        prompts = [f"A cute photo of a {breed}, high quality, highly detailed, elegant, sharp focus"]
        keys = ["pet"]

        print(f"-------------------------------------\nGenerating {images_to_generate} images for breed {breed}...\n")
        start_time = time.time()

        # Images generation loop
        for image_name in controlNet_sample_images:
            
            # Clear memory
            torch.cuda.empty_cache() 

            # Generate images using the Stable Diffusion model conditioned on controlNet
            generate_image(model_path, prompts, keys, path_to_dataset, image_name)

            # Move generated images to the corresponding folder and create annotations
            str_annotations = ""
            dst = f'{path_to_dataset}/images'
            current_time = datetime.now().strftime("%H:%M:%S")
            generated_images_path = "/zhome/d1/6/191852/MSc-thesis/data/generated_images"

            for i, filename in enumerate(os.listdir(generated_images_path)):
                
                # Check if file is not empty - NSFW images are empty
                if os.stat(os.path.join(generated_images_path, filename)).st_size > 1000:

                    # Rename generated image
                    file_path = os.path.join(generated_images_path, filename)
                    image_code = breed + "_controlNet" + "_" + str(i) + "_" + str(current_time)
                    file_name = generated_images_path + "/" + image_code + ".jpg"
                    os.rename(file_path, file_name)

                    try:
                        # Create a copy of the original xml file
                        original_file = f'{path_to_dataset}/annotations/xmls/{image_name}.xml'
                        copy_file = f'{path_to_dataset}/annotations/xmls/{breed}_controlNet_{i}_{current_time}.xml'
                        shutil.copyfile(original_file, copy_file)

                        # Create a copy of the trimap file 1
                        original_file = f'{path_to_dataset}/annotations/trimaps/._{image_name}.png'
                        copy_file = f'{path_to_dataset}/annotations/trimaps/._{breed}_controlNet_{i}_{current_time}.png'
                        shutil.copyfile(original_file, copy_file)

                        # Create a copy of the trimap file 2
                        original_file = f'{path_to_dataset}/annotations/trimaps/{image_name}.png'
                        copy_file = f'{path_to_dataset}/annotations/trimaps/{breed}_controlNet_{i}_{current_time}.png'
                        shutil.copyfile(original_file, copy_file)
                    except: 
                        pass

                    # Move generated image to the corresponding folder
                    shutil.copy(file_name, dst)

                    # Create annotations
                    str_annotations = str_annotations + "\n" + image_code + f" {class_by_id[breed]}" + f" {species_by_id[breed]}" + f" {breed_by_id[breed]}" 

            # Delete all files in generated_images_path
            pipeline_utils.delete_files(generated_images_path)

            # Add annotations to the generated images
            with open(f'{path_to_dataset}/annotations/trainval.txt', 'a') as file:
                file.write(str_annotations)


        # Time elapsed
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds\n")