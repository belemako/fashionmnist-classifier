import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from train import model  # Reuse model architecture

# Classes
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load a single image from test set
transform = transforms.ToTensor()
testset = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

image, label = testset[0]

# Load model weights
model.load_state_dict(torch.load("models/fashion_model.pth"))
model.eval()

# Prediction
with torch.no_grad():
    output = model(image.unsqueeze(0))
    predicted = torch.argmax(output).item()

# Display
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Prediction: {classes[predicted]}\nActual: {classes[label]}")
plt.axis('off')
plt.show()
