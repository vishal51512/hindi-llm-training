import torch

from causal_attention import SelfAttention

x = torch.randn(2, 8, 256)

attention = SelfAttention(
    d_model=256
)

output = attention(x)

print(output.shape)
