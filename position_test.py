import torch
import torch.nn as nn

seq_len = 8
d_model = 256

position_embedding = nn.Embedding(
    seq_len,
    d_model
)

positions = torch.arange(seq_len)

output = position_embedding(positions)

print(output.shape)
