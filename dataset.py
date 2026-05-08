import torch
from torch.utils.data import Dataset
import sentencepiece as spm


class HindiDataset(Dataset):

    def __init__(self,
                 file_path,
                 tokenizer_path,
                 seq_len=128,
                 max_chars=10000):

        self.seq_len = seq_len

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read(max_chars)

        self.tokens = self.sp.encode(text)

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):

        chunk = self.tokens[idx:idx+self.seq_len+1]

        x = torch.tensor(chunk[:-1])

        y = torch.tensor(chunk[1:])

        return x, y
