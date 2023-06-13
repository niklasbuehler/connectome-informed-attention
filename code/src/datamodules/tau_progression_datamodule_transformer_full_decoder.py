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
            x = data.loc[:len(data) - 2].to_numpy().astype(np.float32)
            y = data.loc[len(data) - 1].iloc[2:202].to_numpy().astype(np.float32)
            self.sequences.append((x, y))

    def __getitem__(self, index):
        sequence = self.sequences[index][0]
        decoder_tgt = sequence[-1, :].reshape(1, 203)
        target = self.sequences[index][1]

        return torch.from_numpy(sequence), torch.from_numpy(decoder_tgt), torch.from_numpy(target)

    def __len__(self):
        return len(self.sequences)



class TauProgressionDataModuleTransformerFullDecoder(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            dataset_filename = "tau_progression_sequences.csv",
            split_filename = "train_val_test_split.json",
            batch_size: int = 64,
            max_len: int = None,
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
            collate_fn=(lambda x: collate(x, max_len=self.hparams.max_len)),
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=(lambda x: collate(x, max_len=self.hparams.max_len)),
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
            collate_fn=(lambda x: collate(x, max_len=self.hparams.max_len)),
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )

def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def collate(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - decoder_tgt: torch tensor of shape (1, feat_dim)
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    if batch_size>1:
        features, decoder_tgts, labels = zip(*data)
    else:
        features, decoder_tgts, labels = data[0]
        features, decoder_tgts, labels = [features], [decoder_tgts], [labels]


    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)
    decoder_tgts = torch.stack(decoder_tgts, dim=0)
    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, decoder_tgts, targets, padding_masks

if __name__ == '__main__':
    ds = TauProgressionDataModuleTransformer(data_dir="/vol/chameleon/users/derbel/connectome-based-tau-spread-prediction/data", dataset_filename="tau_progression_sequences_test.csv", split_filename= "train_test_split_test.json", max_len=11, batch_size=1)
    ds.setup(stage="test")
    dl = ds.test_dataloader()
    lens = []
    for x in dl:
        lens.append(x[2].shape[1])
