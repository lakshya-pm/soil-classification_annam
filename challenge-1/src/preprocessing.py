"""

Author: Lakshya Marwaha
Team Name: Individual
Team Members: -
Leaderboard Rank: 48

"""

import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Label encoding for multi-class and binary tasks
def encode_labels(labels, task='multiclass'):
    if task == 'multiclass':
        label_map = {'Alluvial soil': 0, 'Black soil': 1, 'Red soil': 2, 'Clay soil': 3}
        return labels.map(label_map)
    elif task == 'binary':
        # 1 for soil, 0 for non-soil (adjust as per your binary task definition)
        return labels.apply(lambda x: 1 if x == 'Soil' else 0)
    else:
        raise ValueError('Unknown task type')

# Albumentations transforms for training and validation
def get_transforms(img_size=224, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2(),
        ])

class SoilDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, label_col='label', task='multiclass'):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_col = label_col
        self.task = task
        self.encoded_labels = encode_labels(self.df[self.label_col], task)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        label = self.encoded_labels.iloc[idx]
        return image, label

# Utility to create DataLoader
def get_dataloader(df, img_dir, batch_size=32, is_train=True, label_col='label', task='multiclass', img_size=224):
    transforms = get_transforms(img_size, is_train)
    dataset = SoilDataset(df, img_dir, transforms, label_col, task)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=2)

def preprocessing():
    print("This is the file for preprocessing")
    return 0
