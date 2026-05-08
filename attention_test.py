import torch
import torch.nn.functional as F
import math

# Example input
x = torch.randn(1, 4, 8)

# Create Q, K, V
Q = x
K = x
V = x

# Attention scores
scores = Q @ K.transpose(-2, -1)

# Scale scores
scores = scores / math.sqrt(x.size(-1))

# Softmax
weights = F.softmax(scores, dim=-1)

# Final attention output
output = weights @ V

print("Scores Shape:", scores.shape)

print("Weights Shape:", weights.shape)

print("Output Shape:", output.shape)
