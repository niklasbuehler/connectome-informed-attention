import torch
from torch import nn
import pytorch_lightning as pl


class Classifier(pl.LightningModule):
    def __init__(self, lr: float = 0.000002):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(200, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 3))

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.net(x)
        loss = self.criterion(z, y)
        self.log('train_loss', loss)
        return loss
        # --------------------------

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.net(x)
        loss = self.criterion(z, y)
        self.log('val_loss', loss)
        # --------------------------

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.net(x)
        loss = self.criterion(z, y)
        self.log('test_loss', loss)
        # --------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
