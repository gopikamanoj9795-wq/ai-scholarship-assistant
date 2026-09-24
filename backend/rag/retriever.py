import sys
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from langchain_community.vectorstores import Chroma

from backend.config import CHROMA_DB_DIR
from backend.rag.ingest import get_embeddings



def get_vector_db() -> Chroma:
    """Load the Chroma vector database from disk."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embeddings,
    )


def retrieve_policy(query: str, scholarship_id: Optional[str] = None, k: int = 3) -> str:
    """
    Performs a semantic search over the Chroma vector database and returns
    the most relevant policy chunk(s) as formatted plain text.
    
    If scholarship_id is provided, filters results to that specific scholarship.
    """
    if not CHROMA_DB_DIR.exists():
        return "No policy documents database found. Please run ingestion first."

    vector_db = get_vector_db()

    filter_dict = None
    if scholarship_id:
        filter_dict = {"scholarship_id": scholarship_id.strip().upper()}

    search_kwargs = {"k": k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    docs = vector_db.similarity_search(query, **search_kwargs)

    if not docs:
        # If filtered search returned nothing, fallback to unfiltered top results if needed
        if filter_dict:
            docs = vector_db.similarity_search(query, k=k)

    if not docs:
        return "No relevant policy information found."

    # Join chunk contents with clean dividers
    return "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)
