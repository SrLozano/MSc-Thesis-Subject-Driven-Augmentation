import cv2
import sys
import torch
import random
import numpy as np
import pipeline_utils 

from PIL import Image
from datetime import datetime

from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel

sys.path.insert(1, '/zhome/d1/6/191852/MSc-thesis') # caution: path[0] is reserved for script path (or '' in REPL)
from visualization import save_images

path_to_dataset = "../../../../../../work3/s226536/datasets/oxford-iiit-pet"
images_to_generate = 3
model_path = "runwayml/stable-diffusion-v1-5"

# Get samples by breed, class, specie and breed id from txt file
samples_by_breed, class_by_id, species_by_id, breed_by_id = pipeline_utils.get_breeds(f'{path_to_dataset}/annotations/trainval.txt')

breeds_to_generate = list(samples_by_breed.keys())


def generate_images_controlNet(prompts, keys):

    # Get a sample of images to apply controlNet to
    random_images = random.sample(samples_by_breed[breed], 5)

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

        # Generate images
        for i in range(len(prompts)):
            images = pipe(prompts[i], image=canny_image, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=1).images
            save_images(images, keys[i], "inference")





# Data augmentation generation for the selected breeds
for breed in breeds_to_generate:


    prompts = [f"A cute photo of a {breed}, high quality, highly detailed, elegant, sharp focus"]
    keys = ["pet"]

    generate_images_controlNet(prompts, keys)
    