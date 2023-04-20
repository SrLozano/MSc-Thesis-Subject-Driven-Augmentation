import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, transforms

DATA_DIR = "/zhome/d1/6/191852/MSc-thesis/oxford-iiit-pet-segmentation"
batch_size = 16

# Define normal transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

training_data = datasets.OxfordIIITPet(root=DATA_DIR, download=True, transform=transform,  target_types="segmentation")
validation_data = datasets.OxfordIIITPet(root=DATA_DIR, download=True, transform=transform, target_types="segmentation", split="test")

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")