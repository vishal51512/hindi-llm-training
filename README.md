# Hindi LLM Training

Minimal GPT-style transformer model and training scripts for Hindi text.

## Model Overview

The main model is `HindiGPT`, a decoder-only Transformer with token and
position embeddings, stacked transformer blocks, and a final language modeling
head.

**Default configuration**
- Vocabulary size: 16,000
- Model width (`d_model`): 256
- Layers: 6
- Attention heads: 8
- Feed-forward hidden size: 1,024
- Sequence length: 128

**Total parameters (default config): 13,012,608**

## Scripts

- `train.py` — trains the model on `data/hindi_corpus.txt`
- `generate.py` — loads `hindi_gpt.pth` and generates text
- `tokenizer_train.py` — trains the SentencePiece tokenizer
- `custom_agent.py` — watches uploaded documents and auto-trains on new files
- `test_model.py` — quick forward pass sanity check

## Basic Usage

1. Train the tokenizer (if needed):
   ```
   python tokenizer_train.py
   ```
2. Train the model:
   ```
   python train.py
   ```
3. Generate text:
   ```
   python generate.py
   ```

## Auto-Train Agent for Uploaded Documents

Run the custom agent to watch an uploads folder and retrain when new files are added:

```bash
python custom_agent.py --uploads-dir uploads --poll-seconds 20
```

Supported upload types:
- `.txt`
- `.pdf` (requires `pypdf`: `pip install pypdf`)
- `.docx`

When a new file is detected, the agent:
1. Extracts text and appends it to `data/hindi_corpus.txt`
2. Re-runs `tokenizer_train.py`
3. Re-runs `train.py`

Run one cycle only:

```bash
python custom_agent.py --run-once
```
