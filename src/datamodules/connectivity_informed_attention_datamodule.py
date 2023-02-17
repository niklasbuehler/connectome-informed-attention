"""
PyTorch dataset for loading Tau progression sequences.
"""
import os
import pandas as pd
import json
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np



class TauProgressionDataset(torch.utils.data.Dataset):
    """
    Dataset for loading Tau progression sequences.
    """
    dataset_path = ""

    def __init__(self, sequence_dataset: pd.DataFrame) -> None:
        super().__init__()

        self.sequences = []
        seq_ids = list(sequence_dataset.SEQ_ID.unique())

        for id in seq_ids:
            data = sequence_dataset[sequence_dataset.SEQ_ID == id].drop(columns=["SEQ_ID"]).reset_index(drop=True)
            if len(data)==2:
                x = data.loc[0].to_numpy().astype(np.float32)
                y = data.loc[1].iloc[2:202].to_numpy().astype(np.float32)
                self.sequences.append((x, y))

    def __getitem__(self, index):
        sequence = self.sequences[index][0]
        target = self.sequences[index][1]

        return torch.from_numpy(sequence), torch.from_numpy(target)

    def __len__(self):
        return len(self.sequences)



class ConnectivityInformedAttentionDatamodule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            dataset_filename = "tau_progression_sequences.csv",
            split_filename = "train_val_test_split.json",
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
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

            self.data_train = TauProgressionDataset(sequence_dataset=train_set)
            self.data_val = TauProgressionDataset(sequence_dataset=val_set)
            self.data_test = TauProgressionDataset(sequence_dataset=test_set)


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
            shuffle=True,
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


if __name__ == '__main__':
    ds = ConnectivityInformedAttentionDatamodule(data_dir="/vol/chameleon/users/derbel/connectome-based-tau-spread-prediction/data", dataset_filename="tau_progression_sequences_test.csv", split_filename= "train_test_split_test.json", max_len=11, batch_size=1)
    ds.setup(stage="test")
    dl = ds.test_dataloader()
    lens = []
    for x in dl:
        print(x[0].shape)

    print(max(lens))