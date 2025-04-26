import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import model  # Reuse the model architecture

# Load test data
transform = transforms.ToTensor()
testset = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Load saved weights
model.load_state_dict(torch.load("models/fashion_model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
