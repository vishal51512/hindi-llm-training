import sentencepiece as spm
import os

os.makedirs("tokenizer", exist_ok=True)

spm.SentencePieceTrainer.train(
    input='data/hindi_corpus.txt',
    model_prefix='tokenizer/hindi',
    vocab_size=16000,
    character_coverage=0.9995,
    model_type='bpe'
)

print("Hindi tokenizer trained!")
