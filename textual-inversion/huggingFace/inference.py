from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Visualization utility to show off the generated images
def plot_images(images, id):
    
    print(f"Saving image {id}...")
    
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.savefig(f"textual-inversion-{id}.png")

model_id = "/zhome/d1/6/191852/hugging-face/diffusers/examples/textual_inversion/saved_model/textual_inversion_cat"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

placeholder_token = "<cat-toy>"

prompt = f"Gandalf as a {placeholder_token} fantasy art drawn by disney concept artists, golden colour, high quality, highly detailed, elegant, sharp focus, concept art, character concepts, digital painting, mystery, adventure"
images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=3).images
plot_images(images, "gandalf")

prompt = f"An evil {placeholder_token}."
images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=3).images
plot_images(images, "evil")

prompt = f"A masterpiece of a {placeholder_token} crying out to the heavens. Behind the {placeholder_token}, an dark, evil shade looms over it sucking the life right out of it"
images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=3).images
plot_images(images, "masterpiece")

prompt = f"A mysterious {placeholder_token} approaches the great pyramids of egypt."
images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=3).images
plot_images(images, "pyramids")