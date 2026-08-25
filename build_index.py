from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.vectorstores import FAISS

from langchain_ollama import OllamaEmbeddings


SUPPORTED_EXTENSIONS = {".md", ".pdf", ".txt"}


def load_documents(policy_dir: Path):
    documents = []
    for path in sorted(policy_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".pdf":
            documents.extend(PyPDFLoader(str(path)).load())
        else:
            documents.extend(TextLoader(str(path), encoding="utf-8").load())
    return documents


def build_index(policy_dir: Path, chunk_size: int, chunk_overlap: int) -> None:
    if not policy_dir.exists():
        policy_dir.mkdir(parents=True)

    documents = load_documents(policy_dir)
    if not documents:
        raise FileNotFoundError(
            f"No .txt, .md, or .pdf policy files found in {policy_dir}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(policy_dir))
    print(f"Indexed {len(documents)} documents into {len(chunks)} chunks.")
    print(f"FAISS index saved to {policy_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the FAISS policy index used by the airline RAG API."
    )
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=Path("data/policies"),
        help="Folder containing .txt, .md, and .pdf policy files.",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk-overlap must be smaller than chunk-size.")

    build_index(args.policy_dir, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()