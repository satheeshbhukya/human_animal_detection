
import os
import json
from PIL import Image
from torch.utils.data import Dataset

HUMAN_KEYWORDS = [
    "person", "man", "woman", "boy", "girl", "human"
]

ANIMAL_KEYWORDS = [
    "animal", "dog", "cat", "horse", "sheep", "pig",
    "goat", "rabbit", "lion", "fox", "bat",
    "bird", "fish", "whale", "dolphin", "tortoise",
    "squirrel", "crab", "shellfish", "marine mammal",
    "sea lion", "insect", "giraffe", "elephant"
]

class HumanAnimalClassificationDataset(Dataset):
    """
    0 -> Human
    1 -> Animal
    """

    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, "data")
        self.transform = transform

        with open(os.path.join(root_dir, "labels.json")) as f:
            coco = json.load(f)

        cat_id_to_name = {
            cat["id"]: cat["name"].lower() for cat in coco["categories"]
        }

        image_id_to_name = {
            img["id"]: img["file_name"] for img in coco["images"]
        }

        image_labels = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            cat_name = cat_id_to_name[ann["category_id"]]
            image_labels.setdefault(img_id, set()).add(cat_name)

        self.samples = []
        for img_id, cats in image_labels.items():
            label = None
            if any(any(h in c for h in HUMAN_KEYWORDS) for c in cats):
                label = 0
            elif any(any(a in c for a in ANIMAL_KEYWORDS) for c in cats):
                label = 1

            if label is not None:
                img_path = os.path.join(self.image_dir, image_id_to_name[img_id])
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))

        print(f"Loaded {len(self.samples)} classification samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
