from torch.utils.data import DataLoader
from NDFitter.paths import project_path
from NDFitter.MLP.data.dataset import PointsDataset

def get_dataloader(folder='data/mlp',batch_size=32,split_ratio=0.99):
    folder = project_path(folder)
    train_dataset = PointsDataset(folder,train=True,split_ratio=split_ratio)
    test_dataset = PointsDataset(folder,train=False,split_ratio=split_ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
