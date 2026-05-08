import sentencepiece as spm

sp = spm.SentencePieceProcessor()

sp.load("tokenizer/hindi.model")

text = "भारत एक महान देश है"

tokens = sp.encode(text)

print(tokens)

decoded = sp.decode(tokens)

print(decoded)
