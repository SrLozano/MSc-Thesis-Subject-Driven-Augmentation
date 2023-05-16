# This file contains utilities relating visualization and plotting of the experimental results.

# Import dependencies
import matplotlib.pyplot as plt
from datetime import datetime

# Visualization utility to show off the generated images
def plot_images(images, id, technique="inference", verbose=False):
    
    if verbose:
        print(f"Saving image {id} as a plot...")

    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    current_time = datetime.now().strftime("%H:%M:%S")
    plt.savefig(f"/zhome/d1/6/191852/MSc-thesis/data/generated_images/{technique}-{id}-{current_time}.png")

# Utility for saving the generated pictures
def save_images(images, id, technique="inference", verbose=False):

    if verbose:
        print(f"Saving image {id}...")

    for i in range(len(images)):
        current_time = datetime.now().strftime("%H:%M:%S")
        images[i].save(f"/zhome/d1/6/191852/MSc-thesis/data/generated_images/{technique}-{id}-{i}-{current_time}.png")

# Utility for plotting the results of the experiment 003
def make_experiment_003_plot(x, dict_y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 10))

    # Plot the data iterating over the dictionary
    for key, y in dict_y.items():
        plt.plot(x, y, label=key,  marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x)
    plt.legend(dict_y.keys(), loc='lower right')

    # Save and close the plot
    plt.savefig(f"{filename}.pdf")
    plt.close()

# Utility for plotting the results of the experiment 004
def make_experiment_004_plot(x, dict_y, title, xlabel, ylabel, filename, ylim=False):
    plt.figure(figsize=(10, 10))

    # Plot the data iterating over the dictionary
    for key, y in dict_y.items():
        plt.plot(y[1], y[0], label=key,  marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale('log')
    plt.xticks(x, labels=x)
    if ylim != False: plt.ylim(ylim)
    plt.legend(dict_y.keys(), loc='lower right')

    # Save and close the plot
    plt.savefig(f"{filename}.pdf")
    plt.close()


if __name__ == "__main__":
    
    # Define data for the experiment 03-oxford-iiit-pet
    percentage_of_data = [1, 0.5, 0.25, 0.1, 0.05]

    experiment_003_results = {
        "Baseline": [0.8786, 0.8693, 0.8617, 0.7615, 0.6238],
        "Custom data augmentation": [0.8873, 0.8568, 0.8437, 0.7316, 0.5231],
        "AutoAugment": [0.8835, 0.8671, 0.8361, 0.6908, 0.4855],
        "RandAugment": [0.8884, 0.8731, 0.8524, 0.7523, 0.5487],
        "Stable diffusion prompt": [0.8764, 0.8541, 0.8225, 0.7838, 0.7359],
        "Dreambooth": [0.8568, 0.8415, 0.8105, 0.7419, 0.6869],
        "Textual inversion": [0.8824, 0.8688, 0.8219, 0.7872, 0.7430]
    } 

    make_experiment_003_plot(percentage_of_data, experiment_003_results, "Accuracy of augmentation techniques Vs percentage of data used", "Percentage of data", "Accuracy", "experiment_003")


    # Define data for the experiment 04-generation-percentage-oxford-iiit-pet
    percentage_of_data_004 = [0.05, 0.1, 0.5, 1, 2, 10, 20, 40, 80]

    experiment_004_results = {
        "100% - Stable diffusion prompt": [[0.8829, 0.8845, 0.8764, 0.8726, 0.8573], [0.05, 0.1, 0.5, 1, 2]],
        "100% - Textual inversion": [[0.8796, 0.8802, 0.8824, 0.8853, 0.8644], [0.05, 0.1, 0.5, 1, 2]],
        "100% - Dreambooth": [[0.8807, 0.8709, 0.8568, 0.8535, 0.8454], [0.05, 0.1, 0.5, 1, 2]],
        "5% - Stable diffusion prompt": [[0.7419, 0.7425, 0.7359, 0.6989, 0.7185, 0.7191], [1, 2, 10, 20, 40, 80]],
        "5% - Textual inversion": [[0.6924, 0.7223, 0.7430, 0.7278, 0.7408, 0.7240], [1, 2, 10, 20, 40, 80]],
        "5% - Dreambooth": [[0.6712, 0.7082, 0.7240, 0.7065, 0.7169], [1, 2, 10, 20, 40]]
    } 

    make_experiment_004_plot(percentage_of_data_004, experiment_004_results, "Accuracy of augmentation techniques Vs percentage of data generated", "Percentage of generated data - (log scale)", "Accuracy", "experiment_004", 0.65)


    # Define data for the experiment 03-oxford-iiit-pet
    generated_images = [200, 100, 10, 5]

    experiment_005_results = {
        "Stable diffusion prompt": [0.6908, 0.6548, 0.5868, 0.4589],
        "Dreambooth": [0.6837, 0.6799, 0.5808, 0.4725]
    } 

    make_experiment_003_plot(generated_images, experiment_005_results, "Accuracy of subject-driven techniques Vs number of images generated", "Images generated", "Accuracy", "experiment_005")
