import math
from typing import Any
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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


class TransformerModel(pl.LightningModule):
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
    n_heads : int
        The number of heads in the multiheadattention models.
    n_layers : int
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

    def __init__(self, d_in: int, d_model: int, d_hid: int, d_out: int, n_heads: int, n_layers: int, lr: float,
                 dropout: float = 0.5):
        super().__init__()

        self.model_type = 'Transformer'
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.lr = lr

        self.encoder = nn.Linear(d_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(d_model, d_out)
        self.criterion = nn.MSELoss()

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
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
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

    def step(self, batch: Any):
        x, y, lengths = batch["data"], batch["label"], batch["lengths"]
        x = x.permute(1, 0, 2)
        src_mask = generate_square_subsequent_mask(x.size(0))
        logits = self.forward(x.cuda(), src_mask.cuda())
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}