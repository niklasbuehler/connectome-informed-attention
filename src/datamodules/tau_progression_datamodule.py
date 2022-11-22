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



class TauProgressionDataset(torch.utils.data.Dataset):
    """
    Dataset for loading Tau progression sequences.
    """
    dataset_path = ""

    def __init__(self, split, dataset_path='data/', dataset_filename= "tau_progression_sequences.csv") -> None:
        super().__init__()
        assert split in ['train', 'test', 'val']

        #read subject-level dataset splits
        with open(os.path.join(dataset_path, "train_val_test_split.json"), "r") as f:
            train_val_test_split = json.load(f)

        #get subject ids in the current split set
        subject_ids = train_val_test_split[split]

        df = pd.read_csv(os.path.join(dataset_path, dataset_filename))
        df = df[(df.ID.isin(subject_ids))].drop(columns=["ID", "ses"])
        self.sequences = []

        seq_ids = list(df.SEQ_ID.unique())

        for id in seq_ids:
            data = df[df.SEQ_ID == id].drop(columns=["SEQ_ID"]).reset_index(drop=True)
            x = data.loc[:len(data) - 2].to_numpy()
            y = data.loc[len(data) - 1].iloc[3:204].to_numpy()
            self.sequences.append((x, y))





    def __getitem__(self, index):
        sequence = self.sequences[index][0]
        target = self.sequences[index][1]

        return {
            "data": torch.from_numpy(sequence),
            "label": torch.from_numpy(target)
        }

    def __len__(self):
        return len(self.sequences)



class TauProgressionDataModule(LightningDataModule):
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
            self.data_train = TauProgressionDataset(dataset_path=self.hparams.data_dir, split="train")
            self.data_test = TauProgressionDataset(dataset_path=self.hparams.data_dir, split="test")
            self.data_val = TauProgressionDataset(dataset_path=self.hparams.data_dir, split="val")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=collate,
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
            collate_fn=collate,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )

def collate(batch):
    """
        To be passed to DataLoader as the `collate_fn` argument
    """
    assert isinstance(batch, list)
    data = pad_sequence([b['data'] for b in batch])
    lengths = torch.tensor([len(b['data']) for b in batch])
    label = torch.stack([b['label'] for b in batch])
    return {
        'data': data,
        'label': label,
        'lengths': lengths
    }

if __name__ == '__main__':
    ds = TauProgressionDataModule(data_dir="/vol/chameleon/users/derbel/connectome-based-tau-spread-prediction/data")
    ds.setup(stage="train")
    dl = ds.train_dataloader()
    for x in dl:
        print(x["data"].shape)
        print(x["label"].shape)
        break