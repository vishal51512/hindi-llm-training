import torch

from multihead_attention import MultiHeadAttention

x = torch.randn(2, 8, 256)

model = MultiHeadAttention(
    d_model=256,
    num_heads=8
)

output = model(x)

print("Output Shape:", output.shape)
