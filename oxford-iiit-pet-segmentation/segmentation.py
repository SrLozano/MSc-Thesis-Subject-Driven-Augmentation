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
import torchvision.transforms.functional as TF 


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


class UNET(nn.Module):
    
    def __init__(self, in_channels=3, classes=1):
        super(UNET, self).__init__()
        self.layers = [in_channels, 64, 128, 256, 512, 1024]
        
        self.double_conv_downs = nn.ModuleList(
            [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])
        
        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
             for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])
            
        self.double_conv_ups = nn.ModuleList(
        [self.__double_conv(layer, layer//2) for layer in self.layers[::-1][:-2]])
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

        
    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv
    
    def forward(self, x):
        # down layers
        concat_layers = []
        
        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)
        
        concat_layers = concat_layers[::-1]
        
        # up layers
        for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])
            
            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)
            
        x = self.final_conv(x)
        
        return x 


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

        self.network = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)

        # Freeze all the layers except the last one
        if freeze_layers == True:
            for param in self.network.parameters():
                param.requires_grad = False

        # Replace last layer. Parameters of newly constructed modules have requires_grad=True by default
        self.network.classifier = DeepLabHead(2048, num_classes)



    def forward(self, xb):
        return self.network(xb)


''''def train(dataloader, model, loss_fn, optimizer):
    """
    This function trains a model with the given parameters and dataset
    :param dataloader: Training dataloader
    :param model: Model to train
    :param loss_fn: Loss function
    :param optimizer: Optimizer
    :return: Training loss
    """

    # Set model to training mode
    model.train()
    
    # Compute loss for each batch and update model parameters
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
    
        # Set the gradients of all the parameters in the neural network to zero
        optimizer.zero_grad()

        loss_record = []

        # Compute the gradients of the loss function for each image in the batch 
        for i in range(0, len(y)):
            loss = loss_fn(pred[i], y[i])
            loss_record.append(loss.item())
            loss.backward(retain_graph=True)
        loss = loss_fn(pred, y)
        loss.backward()
        # Update the parameters of the neural network based on the computed gradients
        optimizer.step()

        # Print progress
        if batch % 100 == 0:
            size = len(dataloader.dataset)
            current = (batch + 1) * len(X)
            print(f"Training loss: {np.array(loss_record).mean():>7f} [{current}/{size}]")
    
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

    return loss.item(), correct/total'''




DATA_DIR = "/zhome/d1/6/191852"
batch_size = 4
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
#model = PetsModelSegmentation(num_classes=1).to(device)
model = UNET(in_channels=3, classes=3).to(device)


# Print model if verbose
if verbose: print(model)

# Define loss function, optimizer and early stopper
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
loss_function = nn.CrossEntropyLoss(ignore_index=255)

def train(data, model, optimizer, loss_fn):
    print('Entering into train function')
    loss_values = []    
    for index, batch in enumerate(data):         
        X, y = batch        
        X, y = X.to(device), y.to(device)        
        preds = model(X)  
        
        print(y[0])
        print(np.unique(np.array(y[0].cpu())))
        y = torch.squeeze(y)
        print(y.shape)
        print(np.unique(np.array(y[0].cpu())))

        loss = loss_fn(preds, y)        
        optimizer.zero_grad()        
        loss.backward()        
        optimizer.step()  
        print(loss.item())

        '''transform = transforms.ToPILImage()
        img = transform(y[0]).convert("L")
        plt.figure(figsize=(20, 20))
        plt.imshow(img)
        plt.savefig(f"test.png")'''

    return loss.item()

'''# Compute loss for each batch and update model parameters
for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)
    pred = model(X)
    
    # Visualize segmentation maps of pred
    transform = transforms.ToPILImage()
    img = transform(pred[0].softmax(dim=1)).convert("L")
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.savefig(f"test.png")

    break'''

image = Image.open(f"{DATA_DIR}/oxford-iiit-pet/annotations/trimaps/Abyssinian_1.png").convert("L")
image_np = np.array(image)
print(np.unique(image_np))

print("---------------------")

# Training loop of the model
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, optimizer, loss_fn)
    #train(train_dataloader, model, loss_fn, optimizer)
    # validate(validation_dataloader, model, loss_fn)'''








'''
Visualize segmentation maps of pred
print(type(pred['out'][0].softmax(dim=1)))
    print(pred['out'][0].softmax(dim=1))
    transform = transforms.ToPILImage()
    img = transform(pred['out'][0].softmax(dim=1)).convert("L")
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.savefig(f"test.png")
'''