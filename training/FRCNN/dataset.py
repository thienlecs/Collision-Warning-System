import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from training.FRCNN.config import DATA_DIR
from torchvision import transforms as T

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = T.functional.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1) < self.p:
            image = T.functional.hflip(image)
            if "boxes" in target:
                boxes = target["boxes"]
                img_width = image.width if isinstance(image, Image.Image) else image.shape[-1]
                boxes[:, [0, 2]] = img_width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target

def get_transform(train=True):
    if train:
        return Compose([
            RandomHorizontalFlip(p=0.5),
            ToTensor()
        ])
    else:
        return Compose([
            ToTensor()
        ])

class KITTIDataset(Dataset):
    def __init__(self, split='training', transforms=None):
        self.root = DATA_DIR
        self.split = split
        self.transforms = transforms

        self.image_dir = os.path.join(self.root, split, 'image_2')
        self.label_dir = os.path.join(self.root, split, 'label_2')

        self.images = sorted([f[:-4] for f in os.listdir(self.image_dir) if f.endswith('.png')])
        # Separate vehicle classes
        self.class_to_idx = {'Car': 1, 'Van': 2, 'Truck': 3, 'Pedestrian': 4, 'Cyclist': 5}
        
    def __getitem__(self, idx):
        img_id = self.images[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        label_path = os.path.join(self.label_dir, f"{img_id}.txt")

        img = Image.open(img_path).convert("RGB")
        width, height = img.size
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

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32) 
            area = torch.empty((0,), dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)