import math
from typing import Any, Optional
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from src.utils.lr_scheduler import Scheduler
from torch import nn, Tensor
from torchmetrics.classification.accuracy import Accuracy
from torch.nn import TransformerEncoder, BatchNorm1d, TransformerEncoderLayer, Dropout, Linear, MultiheadAttention
import pandas as pd
from torch.nn.functional import normalize

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    """
    The Positional Embedding module adds a constant term to each element of the input.
    This term is dependent on the position and encodes it as frequencies.

    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

    Attributes
    ----------
    d_model : int
        The input and output dimension of this layer.
    dropout : float
        The dropout value.
    max_len : int
        The maximum sequence length.

    Methods
    -------
    forward(x)
        Adds the positional encoding to the input and applies dropout.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TransformerModelFullConnFusionMulti(pl.LightningModule):
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

        #Transformer encoder
        self.input_encoder = nn.Linear(
            in_features=d_in*4,
            out_features=d_model
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerBatchNormEncoderLayer(
            d_model=d_model,
            nhead=n_encoder_heads,
            dim_feedforward=d_hid,
            dropout=transformer_dropout,
            activation=activation
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=n_encoder_layers
        )

        self.linear_mapping = nn.Linear(
            in_features=(d_model*max_len),
            out_features=200
        )

        # Connectome Layers
        self.connectome_enc_layer = nn.Linear(
            in_features=203,
            out_features=203
        )
        self.connectome_enc_layer.requires_grad_(False)

        self.activation = _get_activation_fn(activation)

        self.dropout = nn.Dropout(dropout)

        self.criterion = nn.MSELoss()
        self.test_acc = Accuracy()
        self.val_acc = Accuracy()

        self.connectome = self.create_connectome_matrix()

        self.init_weights()

    def create_connectome_matrix(self):
        #mat = torch.tensor(pd.read_csv("/u/home/bue/Documents/connectome-based-tau-spread-prediction/data/Connectome_mean_rs3scaled_1065.csv", header=None).values)
        mat = torch.tensor(pd.read_csv("/u/home/bue/Documents/connectome-based-tau-spread-prediction/data/N69_HC_functional_connectivity.csv", header=None).values)
        # extend with identity matrix
        mat = torch.cat((torch.zeros(3, 200), mat), 0)
        mat = torch.cat((torch.zeros(203, 3), mat), 1)
        mat = normalize(mat)
        mat = mat + torch.eye(203, 203)
        return mat.float()

    def init_weights(self) -> None:
        initrange = 0.1
        self.connectome_enc_layer.weight.data = self.connectome
        self.connectome_enc_layer.bias.data.zero_()

        self.input_encoder.weight.data.uniform_(-initrange, initrange)
        self.linear_mapping.bias.data.zero_()
        self.linear_mapping.weight.data.uniform_(-initrange, initrange)

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
        conn_enc1 = self.connectome_enc_layer(src) # (seq_len, batch_size, d_in)
        conn_enc2 = self.connectome_enc_layer(conn_enc1) # (seq_len, batch_size, d_in)
        conn_enc3 = self.connectome_enc_layer(conn_enc2) # (seq_len, batch_size, d_in)
        src = torch.cat((src, conn_enc1, conn_enc2, conn_enc3), dim=2) # (seq_len, batch_size, d_in*4)

        src = self.input_encoder(src) * math.sqrt(self.d_model) # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src) # (seq_len, batch_size, d_model)
        output = self.transformer_encoder(src, src_key_padding_mask=~padding_masks) # (seq_len, batch_size, d_model)
        output = self.activation(output)
        output = output.permute(1, 0, 2) # (batch_size, seq_len, d_model)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * padding_masks.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1) # (batch_size, seq_len*d_model)
        output = self.linear_mapping(output) # (batch_size, d_out)
        return output

    def step(self, batch: Any):
        x, y, masks = batch
        x = x.permute(1, 0, 2)
        src_mask = generate_square_subsequent_mask(x.size(0)).to(x.get_device())
        logits = self.forward(x, src_mask, masks, self.connectome.to(x.get_device()))
        loss = self.criterion(logits, y)
        return loss, logits, y

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
        #scheduler = Scheduler(optimizer=optimizer, dim_embed= self.d_hid,warmup_steps=1000, verbose=True)
        scheduler = Scheduler(optimizer=optimizer, dim_embed= self.d_hid,warmup_steps=0, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        #return {"optimizer": optimizer}



def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))