import torch

from diffusers import StableDiffusionPipeline
from visualization import plot_images, save_images

prompts = ["A mysterious golden retriever approaches the great pyramids of egypt",
           "A cute photo of a golden retriever, golden colour, high quality, highly detailed, elegant, sharp focus",
           "A realistic picture of a golden retriever on the moon, 4k, detailed, vivid colors",
           "A cute photo of a golden retriever, high quality, highly detailed, elegant, sharp focus"
          ]
keys = ["pyramids", "golden", "moon", "quality"]

model_id = "runwayml/stable-diffusion-v1-5"  
number_images = 3

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

for i in range(len(prompts)):
    images = pipe(prompts[i], num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=number_images).images
    save_images(images, keys[i], "inference")