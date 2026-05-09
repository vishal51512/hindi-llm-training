import torch

from transformer_block import TransformerBlock

x = torch.randn(2, 8, 256)

block = TransformerBlock(
    d_model=256,
    num_heads=8,
    hidden_dim=1024,
    seq_len=256
)

output = block(x)

print("Output Shape:", output.shape)
