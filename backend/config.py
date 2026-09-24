from pathlib import Path

# Application Metadata
APP_NAME = "AI Scholarship Assistant"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = "Backend API for student scholarship matching and management."

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
RAG_DIR = BASE_DIR / "backend" / "rag"
CHROMA_DB_DIR = RAG_DIR / "chroma_db"

STUDENTS_FILE = DATA_DIR / "students.json"
SCHOLARSHIPS_FILE = DATA_DIR / "scholarships.json"
APPLICATIONS_FILE = DATA_DIR / "applications.json"


# Ollama LLM Settings
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1"
OLLAMA_CREW_MODEL = "llama3.2:1b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

