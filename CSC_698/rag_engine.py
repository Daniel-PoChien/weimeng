import os
import sys

# Set environment variables for chromadb to avoid validation errors with None values
# These are optional settings that chromadb expects but can be empty
os.environ.setdefault('CHROMA_SERVER_HOST', '')
os.environ.setdefault('CHROMA_SERVER_HTTP_PORT', '')
os.environ.setdefault('CHROMA_SERVER_GRPC_PORT', '')
os.environ.setdefault('CLICKHOUSE_HOST', '')
os.environ.setdefault('CLICKHOUSE_PORT', '')

# Compatibility shim for chromadb + pydantic v2
# Chromadb tries to import BaseSettings from pydantic, but it moved to pydantic-settings in v2
# We need to patch this BEFORE chromadb is imported
_PYDANTIC_SETTINGS_AVAILABLE = False
try:
    import pydantic_settings
    _PYDANTIC_SETTINGS_AVAILABLE = True
    # Patch pydantic module to include BaseSettings before chromadb imports it
    import pydantic
    # Store original __getattr__ if it exists
    _original_getattr = getattr(pydantic, '__getattr__', None)
    
    # Add BaseSettings directly to pydantic module's __dict__ so it's available immediately
    pydantic.__dict__['BaseSettings'] = pydantic_settings.BaseSettings
    
    # Also override __getattr__ to return BaseSettings if requested
    def _patched_getattr(name):
        if name == 'BaseSettings':
            return pydantic_settings.BaseSettings
        if _original_getattr:
            return _original_getattr(name)
        raise AttributeError(f"module 'pydantic' has no attribute '{name}'")
    
    pydantic.__getattr__ = _patched_getattr
except ImportError:
    # pydantic-settings not installed - will show error later
    pass
except Exception:
    # Any other error - continue and let chromadb handle it
    pass

try:
    from langchain_community.document_loaders import CSVLoader
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    CHROMADB_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    CHROMADB_AVAILABLE = False
    error_msg = str(e)
    # Provide helpful error message if pydantic-settings is missing
    if not _PYDANTIC_SETTINGS_AVAILABLE and 'BaseSettings' in error_msg:
        error_msg = (
            "pydantic-settings package is required for chromadb compatibility.\n"
            "Please install it with: pip install pydantic-settings\n"
            f"Original error: {error_msg}"
        )
    IMPORT_ERROR = error_msg

# Use a small, free model that runs on your laptop
if CHROMADB_AVAILABLE:
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
else:
    embedding_function = None

DB_DIR = "./chroma_db"

def build_database():
    if not CHROMADB_AVAILABLE:
        error_msg = f"ChromaDB dependencies not available"
        if IMPORT_ERROR:
            error_msg += f": {IMPORT_ERROR}"
        error_msg += "\nPlease install with: pip install chromadb"
        raise ImportError(error_msg)
    
    print("Building Vector Database... (This happens once)")
    documents = []
    
    # Load all three CSV files
    for file in ["nba_teams.csv", "nba_rosters.csv", "nba_scores.csv"]:
        if os.path.exists(file):
            # FIXED: Added encoding="utf-8" to handle special characters in player names
            loader = CSVLoader(file_path=file, encoding="utf-8")
            documents.extend(loader.load())
            print(f"Loaded {file}")
    
    # Save to ChromaDB
    if documents:
        # Use collection_name for consistency
        Chroma.from_documents(
            documents, 
            embedding_function, 
            persist_directory=DB_DIR,
            collection_name="nba_data"
        )
        print("Database Built Successfully!")
    else:
        print("No CSV files found. Run get_data.py first!")

def get_retriever():
    if not CHROMADB_AVAILABLE:
        error_msg = f"ChromaDB dependencies not available"
        if IMPORT_ERROR:
            error_msg += f": {IMPORT_ERROR}"
        error_msg += "\nPlease install with: pip install chromadb"
        raise ImportError(error_msg)
    
    # Connect to the existing database
    try:
        # Try with collection_name first
        db = Chroma(
            persist_directory=DB_DIR, 
            embedding_function=embedding_function,
            collection_name="nba_data"
        )
    except (AttributeError, Exception) as e:
        # If there's an error, try without collection_name
        try:
            db = Chroma(
                persist_directory=DB_DIR, 
                embedding_function=embedding_function
            )
        except Exception as e2:
            raise RuntimeError(
                f"Error connecting to ChromaDB. This may be a version compatibility issue.\n"
                f"Try: pip install chromadb==0.5.3\n"
                f"Then rebuild: python -c \"from rag_engine import build_database; build_database()\"\n"
                f"Original error: {e2}"
            ) from e2
    
    # Return the top 10 most relevant pieces of data
    return db.as_retriever(search_kwargs={"k": 10})