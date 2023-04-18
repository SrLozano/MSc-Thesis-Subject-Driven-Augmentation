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

# Utility for plotting the results of the experiments
def make_experiment_plot(x, dict_y, title, xlabel, ylabel, filename):
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


if __name__ == "__main__":
    
    # Define data for the experiment 03-oxford-iiit-pet
    percentage_of_data = [1, 0.5, 0.25, 0.1, 0.05]

    experiment_003_results = {
        "Baseline": [0.8786, 0.8693, 0.8617, 0.7615, 0.6238],
        "Custom data augmentation": [0.8873, 0.8568, 0.8437, 0.7316, 0.5231],
        "AutoAugment": [0.8835, 0.8671, 0.8361, 0.6908, 0.4855],
        "RandAugment": [0.8884, 0.8731, 0.8524, 0.7523, 0.5487],
        "Stable diffusion prompt": [0.8764, 0.8541, 0.8225, 0.7838, 0.7359],
        "Dreambooth": [0.8633, 0.8219, 0.7838, 0.7250, 0.6880],
        "Textual inversion": [0.8824, 0.8688, 0.8219, 0.7872, 0.7430]
    } 

    make_experiment_plot(percentage_of_data, experiment_003_results, "Accuracy of augmentation techniques Vs percentage of data used", "Percentage of data", "Accuracy", "experiment_003")
