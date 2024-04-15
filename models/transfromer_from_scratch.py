import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    This class implements positional encoding
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        This function initializes positional encoding for a transformer model in PyTorch.

        :param d_model: The `d_model` parameter typically represents the dimensionality of the model or
        the hidden size of the model. It is a hyperparameter that you need to define based on your
        specific model architecture and requirements. It is commonly used in models like Transformers
        where it represents the dimensionality of the input and output
        :type d_model: int
        :param seq_len: The `seq_len` parameter in the code snippet you provided represents the length
        of the sequence for positional encoding. It is used to create a matrix `pe` of shape `(seq_len,
        d_model)` where positional encodings are calculated based on the position within the sequence.
        The positional encoding matrix is
        :type seq_len: int
        :param dropout: The `dropout` parameter in the `__init__` method of your code snippet is a float
        value representing the dropout rate. In neural networks, dropout is a regularization technique
        where randomly selected neurons are ignored during training. The dropout rate determines the
        probability that a neuron is dropped out or retained during training
        :type dropout: float
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply the sin/cos to even and odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension to this tensor
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # Register this tensor to safe it in the file along with the state of the model
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        This function adds positional encoding to the input tensor and applies dropout before returning
        the result.

        :param x: The parameter `x` is likely a tensor representing the input data to be processed by
        the model. The code snippet you provided seems to be part of a neural network model where
        positional encodings (`self.pe`) are added to the input tensor `x` before passing it through a
        dropout layer
        :return: The forward method is returning the result of applying dropout to the input x after
        adding positional encodings to it.
        """
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps  # needs for numerical stability
        self.alpha = nn.Parameter(torch.ones(1))  # learnable parameter, multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # learnable parameter, added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock:
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        pass


# print(torch.arange(0, 10, dtype=torch.float))
# print(torch.arange(0, 10, dtype=torch.float).unsqueeze(1))
# print(torch.arange(0, 10, 2))
# print(torch.exp(torch.arange(0, 10, 2)))
# print(2.718 ** 2)
# print(torch.exp(torch.tensor(2.7183 * 2)))

# print(torch.tensor([[1, 2, 3], [4, 5, 6]]).shape)
# print(torch.tensor([[1, 2, 3], [4, 5, 6]]).unsqueeze(0).shape)
