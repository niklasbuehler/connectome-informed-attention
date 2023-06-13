import torch
from torch import nn
import pytorch_lightning as pl
from typing import Any
from torchmetrics.classification.accuracy import Accuracy

class Mlp(pl.LightningModule):
    def __init__(self,
                 lr: float=0.0008428993084296779):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(203, 408),
                                 nn.ReLU(),
                                 nn.Linear(408, 816),
                                 nn.ReLU(),
                                 nn.Linear(816, 408),
                                 nn.ReLU(),
                                 nn.Linear(408, 200))

        self.criterion = nn.MSELoss()
        self.test_acc = Accuracy()
        self.lr = lr

    def forward(self, sequence) -> Any:
        sequence = sequence.squeeze(0)
        sequence = sequence.squeeze(1)
        output = self.net(sequence)
        return output
    def step(self, batch: Any):
        x, y, lengths = batch["data"], batch["label"], batch["lengths"]
        x = torch.unsqueeze(x,0)
        logits = self.forward(x)
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
        print(logits.shape)
        print(targets.shape)
        target_tau_positivity = (torch.mean(targets.float(), dim=1)>1.3).int()
        preds_tau_positivity = (torch.mean(logits.float(), dim=1) > 1.3).int()
        acc = self.test_acc(preds_tau_positivity, target_tau_positivity)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "logits": logits, "targets": targets}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
