import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

from data.DataLoaderX import DataLoaderX

class MNISTDataset(Dataset):
    def __init__(self, path, transform=None):
        data = pd.read_csv(path)
        self.data = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32).view(-1, 1, 28, 28)
        self.label = torch.tensor(data.iloc[:, 0].values, dtype=torch.long)
        self.len = len(self.data)
        
        self.transform = transform

    def __getitem__(self, index):
        
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        
        # change label to one-hot encoding
        label = torch.zeros(10)
        label[self.label[index]] = 1
        return x, label

    def __len__(self):
        return self.len

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, test_path, batch_size=4, num_workers=4, train_shuffle=True, val_shuffle=False, test_shuffle=False, train_transform=None, val_transform=None, test_transform=None):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        
        # hyperparameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle
        
        # transforms
        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MNISTDataset(self.train_path, self.train_transform)
            self.val_dataset = MNISTDataset(self.val_path, self.val_transform)
        elif stage == 'test':
            self.test_dataset = MNISTDataset(self.test_path)

    def train_dataloader(self):
        return DataLoaderX(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoaderX(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.val_shuffle,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoaderX(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.test_shuffle,
            num_workers=self.num_workers,
        )
