import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def visualize_segmentation_maps(images_id, DATA_DIR):
    """
    This function takes a list of image ids and displays the image and the corresponding segmentation map.
    :param images_id: A list of image ids.
    :param DATA_DIR: The path to the dataset.
    """

    # Open images
    images = []
    for image_id in images_id:
        images.append(Image.open(f"{DATA_DIR}/oxford-iiit-pet/images/{image_id}.jpg"))
        images.append(Image.open(f"{DATA_DIR}/oxford-iiit-pet/annotations/trimaps/{image_id}.png").convert("L"))

    # Calculate the number of rows and columns for the subplots
    num_rows = len(images) // 2
    num_cols = 2

    # Create a figure with n/2 rows and 2 columns of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5*num_rows))

    # Loop through the image paths and display each image in a separate subplot
    for i in range(len(images)):
        # Calculate the row and column indices for the current subplot
        row_index = i // 2
        col_index = i % 2
        
        # Display the image in the corresponding subplot
        axes[row_index, col_index].imshow(images[i])
        
        if col_index == 0:
            axes[row_index, col_index].set_title("Original image")
        else:
            axes[row_index, col_index].set_title("Segmentation mask")

    # Save figure
    plt.savefig(f"segmentation_training_data.pdf")


class PetsModelSegmentation(nn.Module):
    """
    This class defines the model. In this case, DeepLabv3 - Feauture extraction is used
    """

    def __init__(self, num_classes=3, pretrained=True):
        """
        DeepLabv3 class with custom head. DeepLabv3 model with the ResNet101 backbone
        :param outputchannels: The number of output channels in the dataset masks. Defaults to 1
        """
        super().__init__()

        self.network = models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=True)

        # Freeze all the layers except the last one
        if freeze_layers == True:
            for param in self.network.parameters():
                param.requires_grad = False

        # Replace last layer. Parameters of newly constructed modules have requires_grad=True by default
        self.network.classifier = DeepLabHead(2048, num_classes)

    def forward(self, xb):
        return self.network(xb)

class EarlyStopper:
    """
    This class implements early stopping
    """
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.max_validation_accuracy = 0

    def early_stop(self, validation_accuracy):
        # Check if validation loss has decreased
        if validation_accuracy > self.max_validation_accuracy:
            self.max_validation_accuracy = validation_accuracy
            self.counter = 0
        elif validation_accuracy <= self.max_validation_accuracy:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train(dataloader, model, loss_fn, optimizer):
    """
    This function trains a model with the given parameters and dataset
    :param dataloader: Training dataloader
    :param model: Model to train
    :param loss_fn: Loss function
    :param optimizer: Optimizer
    :return: Training loss
    """

    # Set model to training mode
    size = len(dataloader.dataset)
    model.train()

    # Loop over batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)['out']
        #print(pred)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"loss: {loss.item()}")
        break
        # Print progress
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Training loss: {loss:>7f}  [{current}/{size}]")
    
    # Compute training accuracy
    correct = 0
    total = 0
    with torch.no_grad(): # Speed up computations. No gradients needed.
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)['out']
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Training accuracy: %.2f %%' % (100 * correct / total))

    return loss.item(), correct/total




DATA_DIR = "/zhome/d1/6/191852"
batch_size = 2
verbose = False
epochs = 10
learning_rate = 10e-3
freeze_layers = True

# Define normal transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

training_data = datasets.OxfordIIITPet(root=DATA_DIR, download=True, target_types="segmentation", transform=transform, target_transform=transform)
validation_data = datasets.OxfordIIITPet(root=DATA_DIR, download=True, target_types="segmentation", transform=transform, target_transform=transform, split="test")

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Visualize segmentation maps of random samples
images_id = random.Random(5).sample(os.listdir(f"{DATA_DIR}/oxford-iiit-pet/images") , 3)
images_id = list(map(lambda x: x[:-4], images_id)) # Remove file extension with lambda function
visualize_segmentation_maps(images_id, DATA_DIR)

# Initialize the model for this run
model = PetsModelSegmentation(num_classes=1).to(device)

# Print model if verbose
if verbose: print(model)

# Define loss function, optimizer and early stopper
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
early_stopper = EarlyStopper(patience=5)


print("HEY, HERE WE ARE")

for batch, (X, y) in enumerate(train_dataloader):
    print(type(X))
    print(X.shape)
    print("NOW Y-----------------")
    print(type(y))
    print(y.shape)
    print("MODEL--------------")
    X, y = X.to(device), y.to(device)
    pred = model(X)
    print(pred['out'].shape)
    print(type(pred['out'][0].softmax(dim=1)))
    print(pred['out'][0].softmax(dim=1))
    transform = transforms.ToPILImage()
    img = transform(pred['out'][0].softmax(dim=1)).convert("L")
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.savefig(f"test.png")

    loss = loss_fn(pred['out'][0], y[0])
    print(loss.item())




    break

print("SE ACABO LA TONTERIA-----------")









'''# Training loop of the model
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    validate(validation_dataloader, model, loss_fn)

    # Check if early stop
    if early_stopper.early_stop(aux_test_acurracy): 
        print("Early stopping")            
        break'''





'''image = Image.open(f"{DATA_DIR}/oxford-iiit-pet/annotations/trimaps/Abyssinian_1.png").convert("L")

image_np = np.array(image)

print(np.max(image_np))
print(np.min(image_np))'''