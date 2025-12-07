import os
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Directory on disk where Chroma will store its data
DB_DIR = "./chroma_db"

# Small embedding model that runs locally
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# CSV files we index into Chroma
CSV_FILES = ["nba_teams.csv", "nba_rosters.csv", "nba_scores.csv"]


def build_database():
    """
    Build a persistent Chroma vector database from the NBA CSV files.

    Run this once:
        python -c "from rag_engine import build_database; build_database()"
    """
    print("Building Vector Database... (This happens once)")

    documents = []
    for file in CSV_FILES:
        if os.path.exists(file):
            loader = CSVLoader(file_path=file, encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {file} with {len(docs)} rows")
        else:
            print(f"WARNING: {file} not found.")

    if not documents:
        print("No CSV files found. Run get_data.py first!")
        return

    # Create / overwrite the Chroma database
    _ = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=DB_DIR,
        collection_name="nba_data",
    )

    print("Database Built Successfully!")


def get_retriever():
    """
    Connect to the existing Chroma database and return a retriever.

    app.py uses this to build a RetrievalQA chain.
    """
    if not os.path.exists(DB_DIR):
        raise RuntimeError(
            "ChromaDB not found on disk. Build it first with:\n"
            "  python -c \"from rag_engine import build_database; build_database()\""
        )

    try:
        db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding_function,
            collection_name="nba_data",
        )
    except Exception as e:
        # Fallback: try without collection_name in case of older Chroma behavior
        try:
            db = Chroma(
                persist_directory=DB_DIR,
                embedding_function=embedding_function,
            )
        except Exception as e2:
            raise RuntimeError(
                "Error connecting to ChromaDB. This is likely a version compatibility issue "
                "between chromadb and pydantic.\n\n"
                "Try:\n"
                "  pip install --upgrade \"pydantic>=2.7,<3\" \"pydantic-settings>=2.0,<3\" chromadb\n"
                "Then rebuild the DB:\n"
                "  python -c \"from rag_engine import build_database; build_database()\"\n\n"
                f"Original error: {e2}"
            ) from e2

    # Return retriever: top 10 relevant snippets
    return db.as_retriever(search_kwargs={"k": 10})

