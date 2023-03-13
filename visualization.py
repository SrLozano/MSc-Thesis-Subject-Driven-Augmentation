import matplotlib.pyplot as plt

# Visualization utility to show off the generated images
def plot_images(images, id, technique="inference"):
    
    print(f"Saving image {id} as a plot...")
    
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.savefig(f"{technique}-{id}.png")

# Utility for saving the generated pictures
def save_images(images, id, technique="inference"):

    print(f"Saving image {id}...")

    for i in range(len(images)):
        images[i].save(f"{technique}-{id}-{i}.png")
