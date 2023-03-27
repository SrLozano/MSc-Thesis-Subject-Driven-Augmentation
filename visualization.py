import matplotlib.pyplot as plt
from datetime import datetime

# Visualization utility to show off the generated images
def plot_images(images, id, technique="inference"):
    
    print(f"Saving image {id} as a plot...")

    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    current_time = datetime.now().strftime("%H:%M:%S")
    plt.savefig(f"/zhome/d1/6/191852/MSc-thesis/data/generated_images/{technique}-{id}-{current_time}.png")

# Utility for saving the generated pictures
def save_images(images, id, technique="inference"):

    print(f"Saving image {id}...")

    for i in range(len(images)):
        current_time = datetime.now().strftime("%H:%M:%S")
        images[i].save(f"/zhome/d1/6/191852/MSc-thesis/data/generated_images/{technique}-{id}-{i}-{current_time}.png")
