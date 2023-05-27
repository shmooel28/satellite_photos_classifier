import torch
from torchvision.io import read_image
import os
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

'''state_labels = {
    0:'Israel',
}'''
state_labels = {
    'Israel':0,
    'Lebanon':1
}


class CustomImageDataset(Dataset):
    def __init__(self, main_dir, transform=None, target_transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.labels = []

        # Collect all image paths and their corresponding labels
        for sub_dir in os.listdir(main_dir):
            sub_dir_path = os.path.join(main_dir, sub_dir)
            if sub_dir in state_labels:
                print(sub_dir)
                if os.path.isdir(sub_dir_path):
                    for file in os.listdir(sub_dir_path):
                        if file.endswith(".jpg") or file.endswith(".png"):
                            image_path = os.path.join(sub_dir_path, file)
                            self.images.append(image_path)
                            self.labels.append(state_labels.get(sub_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = read_image(image_path)
        label = self.labels[idx]

        image = image.to(torch.float32)
        resized_tensor = TF.resize(image, (32, 32))
        normalized_tensor = TF.normalize(resized_tensor, mean=0.5, std=0.5)
        return normalized_tensor, label