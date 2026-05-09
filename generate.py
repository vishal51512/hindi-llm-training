import torch
import torch.nn.functional as F
import sentencepiece as spm

from model import HindiGPT


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load tokenizer
sp = spm.SentencePieceProcessor()

sp.load("tokenizer/hindi.model")


# Load model
model = HindiGPT(
    vocab_size=16000,
    d_model=256,
    num_heads=8,
    num_layers=6,
    hidden_dim=1024,
    seq_len=128
)

model.load_state_dict(
    torch.load(
        "hindi_gpt.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# Prompt
prompt = "भारत"

# Encode
tokens = sp.encode(prompt)

tokens = torch.tensor(
    [tokens],
    dtype=torch.long
).to(device)


# Generation settings
max_new_tokens = 50

temperature = 0.8

top_k = 20


with torch.no_grad():

    for _ in range(max_new_tokens):

        x = tokens[:, -128:]

        logits = model(x)

        logits = logits[:, -1, :]

        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        values, indices = torch.topk(
            logits,
            top_k
        )

        probs = F.softmax(values, dim=-1)

        # Sample token
        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        # Convert sampled index
        next_token = indices.gather(
            -1,
            next_token
        )

        # Append token
        tokens = torch.cat(
            [tokens, next_token],
            dim=1
        )


# Decode
generated = sp.decode(
    tokens[0].tolist()
)

print("\nGenerated Text:\n")

print(generated)
