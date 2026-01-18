
import os
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from dataset import HumanAnimalDataset

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
DATA_DIR = "/content/drive/MyDrive/human_animal_detection/datasets/train"  # images + labels.json
MODEL_PATH = "/content/drive/MyDrive/human_animal_detection/models/detector_mobilenet.pth"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Dataset & Dataloader
dataset = HumanAnimalDataset(DATA_DIR, transform=T.ToTensor())
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Model: FasterRCNN with MobileNetV3 + FPN
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
num_classes = len(dataset.cat2label) + 1  # background + all categories
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)
model.to(device)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}")

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print("Detector model saved to:", MODEL_PATH)
