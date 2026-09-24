import sys
import re
from pathlib import Path
from typing import List

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from backend.config import DOCS_DIR, CHROMA_DB_DIR, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_EMBED_MODEL


def get_embeddings() -> OllamaEmbeddings:
    """Initialize Ollama embeddings model."""
    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_HOST,
    )



def extract_scholarship_id(file_name: str) -> str:
    """Extract scholarship ID from filename (e.g. 'SCH001_policy.txt' -> 'SCH001')."""
    match = re.search(r"(SCH\d+)", file_name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "UNKNOWN"


def load_policy_documents() -> List[Document]:
    """Load all .txt policy files from docs directory as LangChain Documents."""
    documents: List[Document] = []
    if not DOCS_DIR.exists():
        print(f"[WARN] Docs directory '{DOCS_DIR}' does not exist.")
        return documents

    txt_files = sorted(list(DOCS_DIR.glob("*.txt")))
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        sch_id = extract_scholarship_id(file_path.name)
        doc = Document(
            page_content=content,
            metadata={
                "scholarship_id": sch_id,
                "source": file_path.name,
            },
        )
        documents.append(doc)
        print(f"[LOAD] Loaded {file_path.name} (Scholarship ID: {sch_id}, {len(content)} characters)")

    return documents


def ingest_policy_docs() -> int:
    """
    Loads all .txt policy documents, splits them into semantic chunks,
    generates embeddings with Ollama, and stores them in local Chroma DB.
    """
    raw_docs = load_policy_documents()
    if not raw_docs:
        print("[ERROR] No documents found to ingest.")
        return 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"[SPLIT] Split {len(raw_docs)} documents into {len(chunks)} chunks.")

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()

    print(f"[EMBED] Generating embeddings and saving to Chroma at '{CHROMA_DB_DIR}'...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )
    print(f"[DONE] Successfully ingested {len(chunks)} chunks into Chroma DB.")
    return len(chunks)


if __name__ == "__main__":
    ingest_policy_docs()
