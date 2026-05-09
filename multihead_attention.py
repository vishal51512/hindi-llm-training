import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model=256,
                 num_heads=8,
                 seq_len=256):

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # QKV projections
        self.query = nn.Linear(d_model, d_model)

        self.key = nn.Linear(d_model, d_model)

        self.value = nn.Linear(d_model, d_model)

        # Final projection
        self.out = nn.Linear(d_model, d_model)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(seq_len, seq_len))
        )

    def forward(self, x):

        B, T, C = x.shape

        # Create QKV
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into heads
        Q = Q.view(B, T, self.num_heads, self.head_dim)

        K = K.view(B, T, self.num_heads, self.head_dim)

        V = V.view(B, T, self.num_heads, self.head_dim)

        # Move heads dimension
        Q = Q.transpose(1, 2)

        K = K.transpose(1, 2)

        V = V.transpose(1, 2)

        # Attention scores
        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        # Apply causal mask
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float('-inf')
        )

        # Softmax
        weights = F.softmax(scores, dim=-1)

        # Attention output
        out = weights @ V

        # Recombine heads
        out = out.transpose(1, 2)

        out = out.contiguous().view(
            B,
            T,
            C
        )

        # Final projection
        out = self.out(out)

        return out
