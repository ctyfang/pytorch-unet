import torch
import hydra
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
import cv2
import numpy as np


class Images(Dataset):
    def __init__(self, root_dir, transform=None, **kwargs):
        self.root_dir = root_dir
        self.transform = transform
        self.size = len(os.listdir(self.root_dir))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, f"{idx}.png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(1, *img.shape)
        img = img.astype(np.float32)/255.0
        return {"image": img, "path": img_path}


class UNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, **kwargs):
        self.label_dir = os.path.join(root_dir, 'label')
        self.img_dir = os.path.join(root_dir, 'image')
        self.transform = transform
        self.target_transform = target_transform
        self.size = len(os.listdir(self.img_dir))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{idx}.png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(1, *img.shape)
        img = img.astype(np.float32)/255.0

        label_path = os.path.join(self.label_dir, f"{idx}.png")
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (388, 388), interpolation=cv2.INTER_NEAREST).reshape(1, 388, 388)
        label = label.astype(np.float32)/255.0
        return {"image": img, "label": label}


class LitUNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./', val_ratio=0.1, batch_size=4, num_workers=1,
                 input_channels=1, n_classes=1, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def setup(self, stage_name=None):
        # Assign train/val datasets for use in dataloaders
        full = UNetDataset(self.data_dir, transform=self.transform, target_transform=self.transform)
        n_train = int(full.size * (1-self.val_ratio))
        n_val = full.size - n_train
        self.train, self.val = random_split(full, [n_train, n_val])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)
