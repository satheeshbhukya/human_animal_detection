
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class HumanAnimalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Load COCO-style JSON annotations
        with open(os.path.join(data_dir, "labels.json")) as f:
            data = json.load(f)

        # Map category_id -> label index (start from 1, 0 is background)
        self.cat2label = {cat["id"]: i+1 for i, cat in enumerate(data["categories"])}

        # Map image_id -> image info
        self.images = {img["id"]: img for img in data["images"]}

        # Group annotations by image_id
        self.image_id_to_ann = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_id_to_ann:
                self.image_id_to_ann[img_id] = []
            self.image_id_to_ann[img_id].append(ann)

        # Keep a list of image ids for indexing
        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_info = self.images[image_id]
        anns = self.image_id_to_ann.get(image_id, [])

        img_path = os.path.join(self.data_dir, "data", img_info["file_name"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat2label[ann["category_id"]])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            img = self.transform(img)

        return img, target


