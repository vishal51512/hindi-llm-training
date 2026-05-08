import torch
import torch.nn as nn

vocab_size = 16000
d_model = 256
seq_len = 8

# Token embeddings
token_embedding = nn.Embedding(
    vocab_size,
    d_model
)

# Position embeddings
position_embedding = nn.Embedding(
    seq_len,
    d_model
)

# Example tokens
tokens = torch.tensor([
    [10, 20, 30, 40, 50, 60, 70, 80]
])

# Create positions
positions = torch.arange(seq_len)

# Get embeddings
token_vectors = token_embedding(tokens)

position_vectors = position_embedding(positions)

# Add them
x = token_vectors + position_vectors

print("Final Shape:", x.shape)

print(x)
