import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/zhome/d1/6/191852/MSc-thesis')

import torch

from diffusers import StableDiffusionPipeline
from visualization import plot_images, save_images

placeholder_token = "<funny-ret>"

prompts = [f"A mysterious {placeholder_token} approaches the great pyramids of egypt",
           f"A cute photo of a {placeholder_token}, golden colour, high quality, highly detailed, elegant, sharp focus",
           f"A realistic picture of a {placeholder_token} on the moon, 4k, detailed, vivid colors"
          ]
keys = ["pyramids", "golden", "moon"]

#model_id = "runwayml/stable-diffusion-v1-5"
model_id = "/zhome/d1/6/191852/saved_model"  
number_images = 3

for i in range(len(prompts)):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    images = pipe(prompts[i], num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=number_images).images

    plot_images(images, keys[i], "inference")
    save_images(images, keys[i], "inference")