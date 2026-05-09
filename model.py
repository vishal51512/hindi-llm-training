import torch
import torch.nn as nn

from transformer_block import TransformerBlock


class HindiGPT(nn.Module):

    def __init__(self,
                 vocab_size=16000,
                 d_model=256,
                 num_heads=8,
                 num_layers=6,
                 hidden_dim=1024,
                 seq_len=256):

        super().__init__()

        self.seq_len = seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        # Position embeddings
        self.position_embedding = nn.Embedding(
            seq_len,
            d_model
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                seq_len=seq_len
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(d_model)

        # Output layer
        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, x):

        B, T = x.shape

        # Create positions
        positions = torch.arange(
            0,
            T,
            device=x.device
        )

        # Token embeddings
        token_embeddings = self.token_embedding(x)

        # Position embeddings
        position_embeddings = self.position_embedding(
            positions
        )

        # Combine embeddings
        x = token_embeddings + position_embeddings

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        # Output logits
        logits = self.lm_head(x)

        return logits
