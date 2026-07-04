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
        anc_digit, neg_digit = self.get_anc_neg_digits()
        anc_count, neg_count = self.digit_count[anc_digit], self.digit_count[neg_digit]
        anc_idx, pos_idx, neg_idx = self.get_indices(anc_count, neg_count)

        anchor   = self.digit_groups[anc_digit].iloc[anc_idx].drop("label").to_numpy().reshape(self.MNIST_SIDE_LENGTH, self.MNIST_SIDE_LENGTH)
        positive = self.digit_groups[anc_digit].iloc[pos_idx].drop("label").to_numpy().reshape(self.MNIST_SIDE_LENGTH, self.MNIST_SIDE_LENGTH)
        negative = self.digit_groups[neg_digit].iloc[neg_idx].drop("label").to_numpy().reshape(self.MNIST_SIDE_LENGTH, self.MNIST_SIDE_LENGTH)

        anchor   = self._to_tensor(anchor)
        positive = self._to_tensor(positive)
        negative = self._to_tensor(negative)

        return anchor, positive, negative

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Cast raw 0-255 pixel values to a normalized [0, 1] float tensor of
        shape (1, H, W), then apply any tensor-based transforms (e.g.
        torchvision.transforms.v2, random rotation, random erasing, ...)."""
        tensor = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
        if self.transforms is not None:
            tensor = self.transforms(tensor)
        return tensor

    def get_indices(self, anc_count, neg_count):
        anc_idx = int(random.uniform(a= 0, b=anc_count))
        pos_idx = int(random.uniform(a= 0, b=anc_count))
        neg_idx = int(random.uniform(a= 0, b=neg_count))
        return anc_idx, pos_idx, neg_idx

    def get_anc_neg_digits(self):
        anc_digit = int(random.uniform(a=0, b=10))
        neg_digit = int(random.uniform(a=0, b=10))
        while neg_digit == anc_digit:
            neg_digit = int(random.uniform(a=0, b=10))

        return anc_digit, neg_digit