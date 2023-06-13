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


class ConnectivityInformedAttention(pl.LightningModule):
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

        #Transformer encoder
        self.input_encoder = nn.Linear(
            in_features=self.hparams.d_in,
            out_features=self.hparams.d_model
        )

        self.self_attn = MultiheadAttention(self.hparams.d_model, self.hparams.nheads, dropout=self.hparams.dropout)


        self.linear_mapping = nn.Linear(
            in_features=(self.hparams.d_model),
            out_features=203
        )

        self.activation = _get_activation_fn(self.hparams.activation)

        self.register_buffer("connectivity", torch.from_numpy(pd.read_csv(self.hparams.connectivity_path).to_numpy().astype(np.float32)))

        self.batchnorm = BatchNorm1d(self.hparams.d_model, eps=1e-5)

        self.dropout = nn.Dropout(self.hparams.dropout)

        self.criterion = nn.MSELoss()

        self.init_weights()



    def init_weights(self) -> None:
        initrange = 0.1
        self.linear_mapping.bias.data.zero_()
        self.linear_mapping.weight.data.uniform_(-initrange, initrange)

    def forward(self, input) -> Tensor:
        """
        Args:
            src: Tensor of shape [max_seq_len, batch_size, d_in]
                The model input.
            src_mask: Tensor of shape [seq_len, seq_len]
                The square attention mask is required because the self-attention layers are only allowed to attend the
                earlier positions in the sequence.

        Returns:
            output: Tensor of shape [seq_len, batch_size, d_out]
        """

        input = self.input_encoder(input) * math.sqrt(self.hparams.d_model)
        #output of shape (1, batch_size, d_model)
        output = self.self_attn(input, input, input)[0]
        #TODO: investigate
        #it is done like this in the encoder, optional
        output = input + self.dropout(output)
        #output = self.dropout(output)
        #output = output.squeeze(0)
        output = self.batchnorm(output)

        output = self.activation(output)
        #output = output.permute(1, 0, 2)
        output = self.linear_mapping(output)
        return output

    def step(self, batch: Any):
        x, y = batch
        after_connectivity = torch.matmul(x[:,2:202], self.connectivity)
        after_connectivity = nn.functional.normalize(after_connectivity)
        x[:, 2:202] = after_connectivity
        logits = self.forward(x)
        loss = self.criterion(logits, x)
        return loss, logits, x

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
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "logits": logits, "targets": targets}

    def test_step(self, batch, batch_idx):
        loss, logits, targets = self.step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "logits": logits, "targets": targets}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.98), eps=1e-9)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
        scheduler = Scheduler(optimizer=optimizer, dim_embed= self.hparams.d_hid, warmup_steps=1000, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}



def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))