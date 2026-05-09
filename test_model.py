import torch

from model import HindiGPT

# Example token input
x = torch.randint(
    0,
    16000,
    (2, 8)
)

model = HindiGPT()

logits = model(x)

print("Logits Shape:", logits.shape)
