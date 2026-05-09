import torch
import torch.nn as nn

from multihead_attention import MultiHeadAttention
from feedforward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(self,
                 d_model=256,
                 num_heads=8,
                 hidden_dim=1024,
                 seq_len=256):

        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            seq_len=seq_len
        )

        # Feed forward network
        self.ffn = FeedForward(
            d_model=d_model,
            hidden_dim=hidden_dim
        )

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        # Attention + residual
        x = x + self.attention(x)

        # LayerNorm
        x = self.norm1(x)

        # FFN + residual
        x = x + self.ffn(x)

        # LayerNorm
        x = self.norm2(x)

        return x
