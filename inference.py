import torch

from diffusers import StableDiffusionPipeline
from visualization import plot_images, save_images

model_id = "runwayml/stable-diffusion-v1-5"  
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A mysterious golden retriever approaches the great pyramids of egypt"
images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=3).images

plot_images(images, "golden", "inference")
save_images(images, "golden", "inference")