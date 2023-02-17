"""
PyTorch dataset for loading Tau progression sequences.
"""
import os
import pandas as pd
import json
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class RegionDataset(torch.utils.data.Dataset):
    """
    Dataset for loading Tau progression sequences.
    """
    dataset_path = ""

    def __init__(self, sequence_dataset: pd.DataFrame, edge_index=None, edge_attr=None) -> None:
        super().__init__()

        self.edge_index = edge_index
        self.edge_attr = edge_attr

        self.sequences = []
        seq_ids = list(sequence_dataset.SEQ_ID.unique())

        for id in seq_ids:
            data = sequence_dataset[sequence_dataset.SEQ_ID == id].drop(columns=["SEQ_ID"]).reset_index(drop=True)
            if len(data)==2:
                x = data.loc[0].to_numpy().astype(np.float32)
                y = data.loc[1].iloc[2:202].to_numpy().astype(np.float32)
                self.sequences.append((x, y))

    def __getitem__(self, index):
        x = torch.from_numpy(self.sequences[index][0])
        shaefer_rois = x[:200].unsqueeze(1)
        global_features = x[200:]
        node_features = torch.cat((shaefer_rois, global_features.repeat(200, 1)), dim=-1)
        target = torch.from_numpy(self.sequences[index][1]).unsqueeze(1)
        return Data(x=node_features, y=target)

    def __len__(self):
        return len(self.sequences)



class RegionDatamodule(LightningDataModule):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)


        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 10

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = pd.read_csv(os.path.join(self.hparams.data_dir, self.hparams.dataset_filename))

            #read subject-level dataset splits
            with open(os.path.join(self.hparams.data_dir, self.hparams.split_filename), "r") as f:
                train_val_test_split = json.load(f)

            #get subject ids in the current split set
            train_ids, val_ids, test_ids = train_val_test_split["train"], train_val_test_split["val"], train_val_test_split["test"]


            train_set, val_set, test_set = (dataset[(dataset.ID.isin(train_ids))].drop(columns=["ID", "ses"]),
                                            dataset[(dataset.ID.isin(val_ids))].drop(columns=["ID", "ses"]),
                                            dataset[(dataset.ID.isin(test_ids))].drop(columns=["ID", "ses"]))

            self.data_train = RegionDataset(sequence_dataset=train_set)
            self.data_val = RegionDataset(sequence_dataset=val_set)
            self.data_test = RegionDataset(sequence_dataset=test_set)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            pin_memory=False,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            pin_memory=False,
            shuffle=True,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            pin_memory=False,
            shuffle=False,
            drop_last=True
        )


if __name__ == '__main__':
    ds = RegionDatamodule(data_dir="/vol/chameleon/users/derbel/connectome-based-tau-spread-prediction/data", dataset_filename="tau_progression_sequences_test.csv", split_filename= "train_test_split_test.json", batch_size=8)
    ds.setup(stage="test")
    dl = ds.test_dataloader()
    lens = []
    for x in dl:
        lens.append(x[2].shape[1])
