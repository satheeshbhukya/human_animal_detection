
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset_cls import HumanAnimalClassificationDataset

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
DATA_DIR = "/content/drive/MyDrive/human_animal_detection/datasets/train"
MODEL_PATH = "/content/drive/MyDrive/human_animal_detection/models/classifier_resnet18.pth"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset & Dataloader
dataset = HumanAnimalClassificationDataset(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# Model: ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # human vs animal
model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training
epochs = 2
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(loader):.4f} | Acc: {acc:.2f}%")

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print("Classifier saved to:", MODEL_PATH)
