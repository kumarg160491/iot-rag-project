import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import cfg

DATA_DIRS = {
    "manual"         : cfg.data.manuals_dir,
    "fault_code"     : cfg.data.fault_codes_dir,
    "maintenance_log": cfg.data.maintenance_logs_dir,
}

def load_documents():
    all_docs = []
    for doc_type, folder in DATA_DIRS.items():
        if not os.path.exists(folder):
            print(f"Folder not found, skipping: {folder}")
            continue
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            docs = []
            try:
                if filename.endswith(".pdf"):
                    docs = PyPDFLoader(filepath).load()
                elif filename.endswith(".txt"):
                    docs = TextLoader(filepath).load()
                elif filename.endswith(".csv"):
                    docs = CSVLoader(filepath).load()
                else:
                    print(f"Skipping unsupported file: {filename}")
                    continue
                for doc in docs:
                    doc.metadata["doc_type"] = doc_type
                    doc.metadata["source"]   = filename
                all_docs.extend(docs)
                print(f"Loaded [{doc_type}]: {filename} ({len(docs)} pages/rows)")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return all_docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = cfg.data.chunk_size,
        chunk_overlap = cfg.data.chunk_overlap,
        separators    = ["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"\nTotal chunks created: {len(chunks)}")
    return chunks

def store_in_chromadb(chunks):
    print(f"\nLoading embedding model: {cfg.ollama.embedding} via Ollama")
    embeddings = OllamaEmbeddings(
        model    = cfg.ollama.embedding,
        base_url = cfg.ollama.base_url,
    )
    print(f"Storing vectors in ChromaDB at: {cfg.chroma.db_path}")
    vectorstore = Chroma.from_documents(
        documents         = chunks,
        embedding         = embeddings,
        collection_name   = cfg.chroma.collection_name,
        persist_directory = cfg.chroma.db_path,
    )
    print(f"Successfully stored {len(chunks)} chunks in ChromaDB")
    return vectorstore

if __name__ == "__main__":
    print("=" * 50)
    print("IoT RAG — Document Ingestion Pipeline")
    print("=" * 50)
    docs = load_documents()
    if not docs:
        print("\n No documents found. Add files to data/ folders and retry.")
    else:
        print(f"\n Total pages/rows loaded: {len(docs)}")
        chunks = chunk_documents(docs)
        store_in_chromadb(chunks)
        print("\n Ingestion complete! Ready for RAG queries.")