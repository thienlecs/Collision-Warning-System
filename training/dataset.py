import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from training.config import DATA_DIR

class KITTIDataset(Dataset):
    def __init__(self, split='training', transforms=None):
        self.root = DATA_DIR
        self.split = split
        self.transforms = transforms

        self.image_dir = os.path.join(self.root, split, 'image_2')
        self.label_dir = os.path.join(self.root, split, 'label_2')

        self.images = sorted([f[:-4] for f in os.listdir(self.image_dir) if f.endswith('.png')])
        # Treat all vehicles as one class
        self.class_to_idx = {'Car': 1, 'Van': 1, 'Truck': 1, 'Pedestrian': 2, 'Cyclist': 3}
        
    def __getitem__(self, idx):
        img_id = self.images[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        label_path = os.path.join(self.label_dir, f"{img_id}.txt")

        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.split()
                    cls_name = data[0]

                    if cls_name in self.class_to_idx:
                        bbox = [float(x) for x in data[4:8]]
                        boxes.append(bbox)
                        labels.append(self.class_to_idx[cls_name])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([0.])

        target = {
            "boxes": boxes, 
            "labels": labels, 
            "image_id": image_id, 
            "area": area, 
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64) # Treat all instances as individual objects (not a crowd)
        }
        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.images)