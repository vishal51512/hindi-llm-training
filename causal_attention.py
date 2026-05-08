import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):

    def __init__(self,
                 d_model=256,
                 seq_len=256):

        super().__init__()

        self.query = nn.Linear(d_model, d_model)

        self.key = nn.Linear(d_model, d_model)

        self.value = nn.Linear(d_model, d_model)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(seq_len, seq_len))
        )

    def forward(self, x):

        B, T, C = x.shape

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(C)

        # Apply causal mask
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float('-inf')
        )

        weights = F.softmax(scores, dim=-1)

        output = weights @ V

        return output
