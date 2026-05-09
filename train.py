import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import HindiDataset
from model import HindiGPT


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", device)


# Dataset
dataset = HindiDataset(
    file_path="data/hindi_corpus.txt",
    tokenizer_path="tokenizer/hindi.model",
    seq_len=128,
    max_chars=100000
)

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)


# Model
model = HindiGPT(
    vocab_size=16000,
    d_model=256,
    num_heads=8,
    num_layers=6,
    hidden_dim=1024,
    seq_len=128
)

model = model.to(device)


# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

# Loss function
criterion = nn.CrossEntropyLoss()


# Training loop
epochs = 3

for epoch in range(epochs):

    print(f"\nEpoch {epoch+1}")

    for step, (x, y) in enumerate(loader):

        x = x.to(device)

        y = y.to(device)

        # Forward pass
        logits = model(x)

        # Reshape for loss
        B, T, C = logits.shape

        loss = criterion(
            logits.view(B*T, C),
            y.view(B*T)
        )

        # Zero gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        if step % 50 == 0:

            print(
                f"Step {step} | Loss: {loss.item():.4f}"
            )

torch.save(
    model.state_dict(),
    "hindi_gpt.pth"
)

print("Model saved!")
