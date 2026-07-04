import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

class MNISTDataset(Dataset):

    def __init__(self, csv_path: str, transforms = None):
        super().__init__()
        
        self.transforms = transforms
        self.df = pd.read_csv(csv_path)
        self.length = self.df.shape[0]
        self.MNIST_SIDE_LENGTH = 28
    
        self.digit_groups = {digit : self.df[self.df["label"] == digit] for digit in list(range(0, 10))}
        self.digit_count = {digit: self.df[self.df["label"] == digit].shape[0] for digit in list(range(0, 10))}
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        is_same_pair = random.random() < 0.5

        if is_same_pair:
            digit = int(random.uniform(a=0, b=10))
            count = self.digit_count[digit]
            idx1, idx2 = self.get_same_class_indices(count)

            img1 = self.digit_groups[digit].iloc[idx1].drop("label").to_numpy().reshape(self.MNIST_SIDE_LENGTH, self.MNIST_SIDE_LENGTH)
            img2 = self.digit_groups[digit].iloc[idx2].drop("label").to_numpy().reshape(self.MNIST_SIDE_LENGTH, self.MNIST_SIDE_LENGTH)
            label = 1.0
        else:
            digit1, digit2 = self.get_anc_neg_digits()
            count1, count2 = self.digit_count[digit1], self.digit_count[digit2]
            idx1 = int(random.uniform(a=0, b=count1))
            idx2 = int(random.uniform(a=0, b=count2))

            img1 = self.digit_groups[digit1].iloc[idx1].drop("label").to_numpy().reshape(self.MNIST_SIDE_LENGTH, self.MNIST_SIDE_LENGTH)
            img2 = self.digit_groups[digit2].iloc[idx2].drop("label").to_numpy().reshape(self.MNIST_SIDE_LENGTH, self.MNIST_SIDE_LENGTH)
            label = 0.0

        img1 = self._to_tensor(img1)
        img2 = self._to_tensor(img2)
        label = torch.tensor(label, dtype=torch.float32)

        return img1, img2, label

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Cast raw 0-255 pixel values to a normalized [0, 1] float tensor of
        shape (1, H, W), then apply any tensor-based transforms (e.g.
        torchvision.transforms.v2, random rotation, random erasing, ...)."""
        tensor = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
        if self.transforms is not None:
            tensor = self.transforms(tensor)
        return tensor

    def get_same_class_indices(self, count):
        idx1 = int(random.uniform(a=0, b=count))
        idx2 = int(random.uniform(a=0, b=count))
        while idx2 == idx1 and count > 1:
            idx2 = int(random.uniform(a=0, b=count))
        return idx1, idx2

    def get_anc_neg_digits(self):
        anc_digit = int(random.uniform(a=0, b=10))
        neg_digit = int(random.uniform(a=0, b=10))
        while neg_digit == anc_digit:
            neg_digit = int(random.uniform(a=0, b=10))

        return anc_digit, neg_digit