import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.classification.accuracy import Accuracy
from torch.nn.utils.rnn import pack_padded_sequence
from torchmetrics import MinMetric
from typing import Any, List

class LSTMRegressor(pl.LightningModule):
    def __init__(self, lr, hidden_size):
        super().__init__()
        self.lr = lr
        self.lstm1 = nn.LSTM(203, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size*2, batch_first=True)
        self.activation = nn.ReLU()

        self.linear = nn.Linear(hidden_size, 200)

        self.criterion = nn.MSELoss()
        self.test_acc = Accuracy()
    def forward(self, sequence, lengths):
        input = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        x, (h, _) = self.lstm1(input)
        #x.data = self.activation(x.data)
        #x, (h, _) = self.lstm2(x)
        output = h
        output = output.squeeze(0)
        output = self.activation(output)
        output = self.linear(output)
        return output


    def step(self, batch: Any):
        x, y, lengths = batch["data"], batch["label"], batch["lengths"]
        logits = self.forward(x, lengths.cpu())
        loss = self.criterion(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, targets = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "logits": logits, "targets": targets}


    def validation_step(self, batch, batch_idx):
        loss, logits, targets = self.step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "logits": logits, "targets": targets}

    def test_step(self, batch, batch_idx):
        loss, logits, targets = self.step(batch)
        target_tau_positivity = (torch.mean(targets.float(), dim=1)>1.3).int()
        preds_tau_positivity = (torch.mean(logits.float(), dim=1) > 1.3).int()
        acc = self.test_acc(preds_tau_positivity, target_tau_positivity)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "logits": logits, "targets": targets}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

