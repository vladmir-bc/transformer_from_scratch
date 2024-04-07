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
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_model)
        pos_encoding = torch.zeros(seq_len, d_model)

        # add_even = 1
        # add_odd = 2

        # self.embedding_pos = torch.where(
        #     self.embedding % 2 == 0, self.embedding + add_even, self.embedding + add_odd
        # )
