import torch

from feedforward import FeedForward

x = torch.randn(2, 8, 256)

ffn = FeedForward(
    d_model=256,
    hidden_dim=1024
)

output = ffn(x)

print("Output Shape:", output.shape)
