import math
from typing import Any, Optional
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from torch.nn import functional as F
from src.utils.lr_scheduler import Scheduler
from torch import nn, Tensor
from torchmetrics.classification.accuracy import Accuracy
from torch.nn import TransformerEncoder, BatchNorm1d, TransformerEncoderLayer, Dropout, Linear, MultiheadAttention
from torch.nn import MaxPool1d
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GATv2Conv
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import pool



class RegionGAT(pl.LightningModule):
    """
    This module is an implementation of the Transformer architecture presented in the paper "Attention is All You Need".
    We don't use token embedding and replace this embedding step with simple feedforward embedding instead.
    As a criterion, MSE Loss is used.

    https://arxiv.org/abs/1706.03762

    Attributes
    ----------
    d_in : int
        The input dimension (here: 203).
    d_model : int
        The dimension of the space the input is embedded into.
        This is also the dimension of the expected features in the encoder/decoder inputs.
    d_hid : int
        The dimension to use in the feedforward network.
    d_out : int
        The output dimension (here: 200).
    n_encoder_heads : int
        The number of heads in the multiheadattention models.
    n_encoder_layers : int
        The number of sub-encoder-layers in the encoder.
    lr : float
        The learning rate.
    dropout : float
        The dropout value.

    Methods
    -------
    init_weights()
        Initializes the weights randomly and sets the biases to zero.
    forward(src, src_mask)
        Performs a forward pass through the model.
    """

    def __init__(self, **kwargs):
        super().__init__()

        #saves hparams to model checkpoint
        self.save_hyperparameters()

        self.hparams.model_type = 'Transformer'

        self.register_buffer("connectivity", torch.from_numpy(pd.read_csv(self.hparams.connectivity_path).to_numpy().astype(np.float32)))

        # define graph structure and edge weights
        edge_index, edge_attr = dense_to_sparse(self.connectivity)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)

        self.conv1 = GATv2Conv(self.hparams.d_in,
                               self.hparams.d_hid,
                               heads=self.hparams.nheads,
                               edge_dim=1)

        self.conv2 = GATv2Conv(self.hparams.d_hid * self.hparams.nheads,
                               self.hparams.d_out,
                               heads=self.hparams.nheads,
                               edge_dim=1)

        self.dropout = nn.Dropout(self.hparams.dropout)
        self.activation = nn.ReLU()
        self.maxpool = MaxPool1d(self.hparams.nheads, 1)
        self.criterion = nn.MSELoss()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x, batch) -> Tensor:
        #x = self.dropout(x)
        x = self.conv1(x, edge_index=self.edge_index, edge_attr=self.edge_attr)
        #x = torch.cat(torch.split(x, 1, dim=-1), dim=-2)
        x = self.activation(x)
        x = self.conv2(x, edge_index=self.edge_index, edge_attr=self.edge_attr)
        x = self.maxpool(x)
        return x

    def step(self, batch: Any):
        x, y = batch.x, batch.y
        logits = self.forward(x, batch.batch)
        loss = self.criterion(logits, y)
        return loss, logits, y

    def infer(self, batch: Any):
        x, y, masks = batch
        x = x.permute(1, 0, 2)
        self.eval()
        with torch.no_grad():
            output = self.forward(x, masks)
        return output

    def training_step(self, batch, batch_idx):
        loss, logits, targets = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "logits": logits, "targets": targets}


    def validation_step(self, batch, batch_idx):
        loss, logits, targets = self.step(batch)
        target_tau_positivity = (torch.mean(targets.float(), dim=1)>1.3).int()
        preds_tau_positivity = (torch.mean(logits.float(), dim=1) > 1.3).int()
        acc = self.val_acc(preds_tau_positivity, target_tau_positivity)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = Scheduler(optimizer=optimizer, dim_embed= self.hparams.d_hid,warmup_steps=1000, verbose=True)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}



def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))