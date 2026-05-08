import torch
import torch.nn as nn

vocab_size = 16000
d_model = 256

embedding = nn.Embedding(
    vocab_size,
    d_model
)

tokens = torch.tensor([
    [10, 20, 30]
])

output = embedding(tokens)

print(output.shape)
