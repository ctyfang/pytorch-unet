import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np


class UNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.label_dir = os.path.join(root_dir, 'label')
        self.img_dir = os.path.join(root_dir, 'image')
        self.transform = transform
        self.target_transform = target_transform
        self.size = len(os.listdir(self.img_dir))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{idx}.png")
        label_path = os.path.join(self.label_dir, f"{idx}.png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(1, *img.shape)
        img = img.astype(np.float32)/255.0

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (388, 388), interpolation=cv2.INTER_NEAREST).reshape(1, 388, 388)
        label = label.astype(np.float32)/255.0
        return {"image": img, "label": label}
