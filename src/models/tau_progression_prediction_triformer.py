import math
from typing import Any, Optional
import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from src.models.tau_progression_prediction_transformer_full import TransformerModelFull
from src.models.tau_progression_prediction_transformer_full_seq import TransformerModelFullSequence
from src.utils.lr_scheduler import Scheduler
from torch import nn, Tensor
from torchmetrics.classification.accuracy import Accuracy
from torch.nn import TransformerEncoder, BatchNorm1d, TransformerEncoderLayer, Dropout, Linear, MultiheadAttention
import pandas as pd
from torch.nn.functional import normalize


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class TriformerModel(pl.LightningModule):
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

    def __init__(self,
                 d_in: int,
                 max_len: int,
                 d_model: int,
                 d_hid: int,
                 d_out: int,
                 n_encoder_heads: int,
                 n_encoder_layers: int,
                 lr: float,
                 activation: str = 'gelu',
                 transformer_dropout: float = 0.5,
                 dropout: float = 0.2):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model_type = 'Transformer'
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.lr = lr
        self.d_hid = d_hid

        # Transformers
        self.data_transformer = TransformerModelFullSequence(d_in, max_len, d_model, d_hid, d_out, n_encoder_heads,
                                                     n_encoder_layers, lr, activation, transformer_dropout, dropout)
        self.conn_transformer = TransformerModelFullSequence(d_in, max_len, d_model, d_hid, d_out, n_encoder_heads,
                                                     n_encoder_layers, lr, activation, transformer_dropout, dropout)
        self.concat_transformer = TransformerModelFull(d_model*2, max_len, d_model, d_hid, d_out, n_encoder_heads,
                                                       n_encoder_layers, lr, activation, transformer_dropout, dropout)

        self.criterion = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.test_acc = Accuracy()
        self.val_acc = Accuracy()

        self.connectome = self.create_connectome_matrix()

    def create_connectome_matrix(self):
        # mat = torch.tensor(pd.read_csv("/u/home/bue/Documents/connectome-based-tau-spread-prediction/data/Connectome_mean_rs3scaled_1065.csv", header=None).values)
        mat = torch.tensor(pd.read_csv(
            "/u/home/bue/Documents/connectome-based-tau-spread-prediction/data/N69_HC_functional_connectivity.csv",
            header=None).values)
        # extend with identity matrix
        mat = torch.cat((torch.zeros(3, 200), mat), 0)
        mat = torch.cat((torch.zeros(203, 3), mat), 1)
        mat = normalize(mat)
        mat = mat + torch.eye(203, 203)
        return mat.float()

    def forward(self, src: Tensor, src_mask: Tensor, padding_masks: Tensor, connectome: Tensor) -> Tensor:
        """
        Args:
            src: Tensor of shape [seq_len, batch_size, d_in]
                The model input.
            src_mask: Tensor of shape [seq_len, seq_len]
                The square attention mask is required because the self-attention layers are only allowed to attend the
                earlier positions in the sequence.

        Returns:
            output: Tensor of shape [seq_len, batch_size, d_out]
        """
        data_out = self.data_transformer(src, src_mask, padding_masks)

        conn = torch.matmul(src, connectome)
        conn_out = self.conn_transformer(conn, src_mask, padding_masks)

        #print("data: ", data_out.shape)
        #print("conn: ", conn_out.shape)
        concat = torch.cat((data_out, conn_out), dim=2) # (batch_size, seq_len, d_model*2)
        #print("concat: ", concat.shape)
        concat = concat.permute(1, 0, 2)
        #print("concat: ", concat.shape)
        concat_out = self.concat_transformer(concat, src_mask, padding_masks)
        output = concat_out

        # output = output.reshape(output.shape[0], -1) # (batch_size, seq_len*d_model)
        #output = self.linear_mapping(output) # (batch_size, d_out)
        return output

    def step(self, batch: Any):
        x, y, masks = batch
        x = x.permute(1, 0, 2)
        src_mask = generate_square_subsequent_mask(x.size(0)).to(x.get_device())
        logits = self.forward(x, src_mask, masks, self.connectome.to(x.get_device()))
        loss = self.criterion(logits, y)
        l1loss = self.l1loss(logits, y)
        return loss, l1loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, l1loss, logits, targets = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "logits": logits, "targets": targets}


    def validation_step(self, batch, batch_idx):
        loss, l1loss, logits, targets = self.step(batch)
        target_tau_positivity = (torch.mean(targets.float(), dim=1)>1.3).int()
        preds_tau_positivity = (torch.mean(logits.float(), dim=1) > 1.3).int()
        acc = self.val_acc(preds_tau_positivity, target_tau_positivity)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "logits": logits, "targets": targets}

    def test_step(self, batch, batch_idx):
        loss, l1loss, logits, targets = self.step(batch)
        target_tau_positivity = (torch.mean(targets.float(), dim=1)>1.3).int()
        preds_tau_positivity = (torch.mean(logits.float(), dim=1) > 1.3).int()
        acc = self.test_acc(preds_tau_positivity, target_tau_positivity)
        self.log("test_l1_loss", l1loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "logits": logits, "targets": targets}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
        scheduler = Scheduler(optimizer=optimizer, dim_embed= self.d_hid,warmup_steps=1000, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        #return {"optimizer": optimizer}



def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))
