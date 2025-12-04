import os
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Use a small, free model that runs on your laptop
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
DB_DIR = "./chroma_db"

def build_database():
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
        Chroma.from_documents(documents, embedding_function, persist_directory=DB_DIR)
        print("Database Built Successfully!")
    else:
        print("No CSV files found. Run get_data.py first!")

def get_retriever():
    # Connect to the existing database
    db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
    # Return the top 3 most relevant pieces of data
    return db.as_retriever(search_kwargs={"k": 10})