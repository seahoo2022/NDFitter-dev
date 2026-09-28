import pickle
import torch
from torch.utils.data import Dataset
import os
from NDFitter.paths import project_path

class PointsDataset(Dataset):
    def __init__(self, folder='data/mlp', train=True, split_ratio=0.99):
        folder = project_path(folder)
        # Load data
        with open(os.path.join(folder,'points.pickle'), 'rb') as f:
            self.data = torch.from_numpy(pickle.load(f)).float()
        with open(os.path.join(folder,'values.pickle'), 'rb') as f:
            self.target = torch.from_numpy(pickle.load(f)).float()
        self.data.requires_grad_()
        # Split data
        split_idx = int(len(self.data) * split_ratio)
        if train:
            self.data = self.data[:split_idx]
            self.target = self.target[:split_idx]
        else:
            self.data = self.data[split_idx:]
            self.target = self.target[split_idx:]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.Tensor([self.target[idx]])
