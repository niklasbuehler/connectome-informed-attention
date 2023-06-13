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

class MultiheadAttention_mod(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, fixed_matrix):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim * num_heads # for heads to be big enough to incorporate and encode the connectivtz matrix we will set embed dimension to 800
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.fixed_matrix = fixed_matrix

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, ((3 * self.embed_dim) - self.head_dim))
        self.connectivity = nn.Linear(input_dim, self.head_dim) #here the important part is that we think about the dimensionality of the connectivtz
        self.o_proj = nn.Linear(self.embed_dim, embed_dim)

        self._reset_parameters()

    @staticmethod
    def scaled_dot_product(q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        # here i add the connectivty information to the network
        self.connectivity.weight = nn.Parameter(self.fixed_matrix)
        self.connectivity.weight.requires_grad = False
        self.connectivity.bias.requires_grad = False

    def forward(self, x, mask=None, return_attention=False):
        seq_length, batch_size, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        q_connectome = self.connectivity(x)
        qkv = torch.cat((q_connectome, qkv), 2)
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims] it should be (seq_len, batch_size, d_model)
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)
        o1 = o.permute(2, 0 , 1)
        o = o.permute(1, 0 , 2)
        return o
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
        connectivity = self.create_connectome_matrix()
        self.self_attn = MultiheadAttention_mod(d_model, d_model, nhead, connectivity)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.connectome = self.create_connectome_matrix()[3:, 3:] # (200, 200)

        # Initialize weights of attention mechanism
        #print("W shape: ", self.self_attn.in_proj_weight.data.shape) # (d_model*3, d_model) = (200*3, 200)
        print("Normalized Connectome: ", self.connectome)
        #rest_of_attn = self.self_attn.in_proj_weight[200:, :]
        #self.self_attn.in_proj_weight.data = torch.cat((self.connectome, rest_of_attn), 0)
        self.self_attn._reset_parameters()
        #print("W new shape: ", self.self_attn.in_proj_weight.data.shape) # (d_model*3, d_model)

    def create_connectome_matrix(self,extend=False):
        wtf = pd.read_csv("/home/andreszapata/devel/connectome-based-tau-spread-prediction/data/connectivity.csv")
        mat = torch.tensor(wtf.values)
        #mat = torch.tensor(pd.read_csv("/home/andreszapata/devel/connectome-based-tau-spread-prediction/data/connectivity.csv", header=None).values)
        #mat = torch.tensor(pd.read_csv("/home/andreszapata/devel/connectome-based-tau-spread-prediction/data/connectivity.csv", header=None).values)
        # extend with identity matrix
        if extend:
            mat = torch.cat((torch.zeros(3, 200), mat), 0)
            mat = torch.cat((torch.zeros(203, 3), mat), 1)
            mat = mat + torch.eye(203, 203)
        mat = normalize(mat)
        return mat.float()

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
        src2 = self.self_attn(src, src, src)[0]
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


class TransformerModelFullConnAtt(pl.LightningModule):
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
            in_features=d_in,
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
            out_features=d_out
        )

        self.activation = _get_activation_fn(activation)

        self.dropout = nn.Dropout(dropout)

        self.criterion = nn.MSELoss()
        self.test_acc = Accuracy(num_classes=3,task="multiclass")
        self.val_acc = Accuracy(num_classes=3, task="multiclass")

        self.connectome = self.create_connectome_matrix(extend=True)

        self.init_weights()

    def create_connectome_matrix(self, extend=False):
        #wtf = pd.read_csv("/home/andreszapata/devel/connectome-based-tau-spread-prediction/data/connectivity.csv", header=None)
        #mat = torch.tensor(wtf.values)
        wtf = pd.read_csv("/home/andreszapata/devel/connectome-based-tau-spread-prediction/data/connectivity.csv")
        mat = torch.tensor(wtf.values)
        # extend with identity matrix
        if extend:
            mat = torch.cat((torch.zeros(3, 200), mat), 0)
            mat = torch.cat((torch.zeros(203, 3), mat), 1)
            mat = mat + torch.eye(203, 203)
        mat = normalize(mat)
        return mat.float()

    def init_weights(self) -> None:
        initrange = 0.1

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
        scheduler = Scheduler(optimizer=optimizer, dim_embed= self.d_hid,warmup_steps=1000, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        #return {"optimizer": optimizer}



def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))