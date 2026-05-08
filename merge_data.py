import os

output_file = "data/hindi_corpus.txt"

os.makedirs("data", exist_ok=True)

with open(output_file, "w", encoding="utf-8") as outfile:

    folder = "train"

    files = os.listdir(folder)

    for i, filename in enumerate(files):

        path = os.path.join(folder, filename)

        try:
            with open(path, "r", encoding="utf-8") as infile:

                text = infile.read().strip()

                if text:
                    outfile.write(text + "\n")

        except Exception as e:
            print("Error:", filename)

        if i % 1000 == 0:
            print(f"Processed {i} files")

print("Dataset merged!")
