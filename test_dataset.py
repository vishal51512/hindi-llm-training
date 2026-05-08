from dataset import HindiDataset

dataset = HindiDataset(
    file_path="data/hindi_corpus.txt",
    tokenizer_path="tokenizer/hindi.model",
    seq_len=8,
    max_chars=5000
)

x, y = dataset[0]

print("Input:", x)
print("Target:", y)
