import argparse
import json
import subprocess
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def extract_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def extract_pdf(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires pypdf. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_docx(file_path: Path) -> str:
    with zipfile.ZipFile(file_path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    text_nodes = root.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
    return " ".join(node.text for node in text_nodes if node.text)


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return extract_txt(file_path)
    if suffix == ".pdf":
        return extract_pdf(file_path)
    if suffix == ".docx":
        return extract_docx(file_path)

    raise ValueError(f"Unsupported file format: {file_path.suffix}")


def load_processed(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    return set(json.loads(state_file.read_text(encoding="utf-8")))


def save_processed(state_file: Path, processed: set[str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps(sorted(processed), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_to_corpus(corpus_path: Path, text: str) -> None:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("a", encoding="utf-8") as corpus:
        corpus.write(text.strip() + "\n")


def run_training_pipeline(project_root: Path) -> None:
    subprocess.run(
        ["python", str(project_root / "tokenizer_train.py")],
        cwd=project_root,
        check=True,
    )
    subprocess.run(
        ["python", str(project_root / "train.py")],
        cwd=project_root,
        check=True,
    )


def process_new_files(
    uploads_dir: Path,
    corpus_path: Path,
    state_file: Path,
    project_root: Path,
) -> bool:
    processed = load_processed(state_file)
    candidates = sorted(
        file_path for file_path in uploads_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    new_files = [file_path for file_path in candidates if str(file_path) not in processed]
    if not new_files:
        return False

    print(f"Found {len(new_files)} new file(s). Updating corpus...")

    for file_path in new_files:
        text = extract_text(file_path).strip()
        if not text:
            print(f"Skipping empty file: {file_path.name}")
            processed.add(str(file_path))
            continue

        append_to_corpus(corpus_path, text)
        processed.add(str(file_path))
        print(f"Added text from: {file_path.name}")

    save_processed(state_file, processed)

    print("Starting tokenizer + model training...")
    run_training_pipeline(project_root)
    print("Training completed.")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Custom auto-train agent for uploaded documents."
    )
    parser.add_argument("--uploads-dir", default="uploads")
    parser.add_argument("--corpus-path", default="data/hindi_corpus.txt")
    parser.add_argument("--state-file", default=".agent_state/processed_files.json")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--run-once", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    uploads_dir = (project_root / args.uploads_dir).resolve()
    corpus_path = (project_root / args.corpus_path).resolve()
    state_file = (project_root / args.state_file).resolve()

    uploads_dir.mkdir(parents=True, exist_ok=True)

    print(f"Watching uploads in: {uploads_dir}")
    print(f"Corpus path: {corpus_path}")
    print(f"Polling every {args.poll_seconds} seconds")

    while True:
        try:
            process_new_files(
                uploads_dir=uploads_dir,
                corpus_path=corpus_path,
                state_file=state_file,
                project_root=project_root,
            )
        except Exception as exc:
            print(f"Agent error: {exc}")

        if args.run_once:
            break

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
