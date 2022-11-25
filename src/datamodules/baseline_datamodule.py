"""
PyTorch dataset for loading snapshots of ModelNet 3D meshes.
"""
import numpy as np
from typing import Optional, Tuple
import torch
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset

classes = {
    "CN": 0,
    "Dementia": 1,
    "MCI": 2
}


class SchaeferDataset(torch.utils.data.Dataset):
    """
    Dataset for loading Schaefer data
    """
    dataset_path = ""

    def __init__(self,  X, y) -> None:
        super().__init__()

        self.X = X
        self.y = torch.tensor([classes[i] for i in y.to_list()])

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]).float(), self.y[index]


    def __len__(self):
        return self.y.shape[0]


## TODO: adapt code for our purposes
class SchaeferDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            data_dir: str = "data/",
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.df = pd.read_csv(data_dir)
        self.data_dir = data_dir
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 3

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            X = self.df.loc[:, 'SUVR.Schaefer200.ROI.idx.1':'SUVR.Schaefer200.ROI.idx.200'].to_numpy()
            y = self.df.loc[:, "Diagnostic"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)
            self.data_train = SchaeferDataset(X_train, y_train)
            self.data_test = SchaeferDataset(X_test, y_test)
            self.data_val = SchaeferDataset(X_val, y_val)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )


