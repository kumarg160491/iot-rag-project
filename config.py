from dataclasses import dataclass

@dataclass(frozen=True)
class OllamaConfig:
    base_url : str = "http://localhost:11434"
    model    : str = "llama3.2"
    embedding: str = "nomic-embed-text"
    temperature: float = 0.1

@dataclass(frozen=True)
class ChromaConfig:
    db_path        : str = "./chroma_db"
    collection_name: str = "iot_knowledge_base"

@dataclass(frozen=True)
class RetrieverConfig:
    search_type: str = "similarity"
    top_k      : int = 4

@dataclass(frozen=True)
class DataConfig:
    manuals_dir         : str = "data/manuals"
    fault_codes_dir     : str = "data/fault_codes"
    maintenance_logs_dir: str = "data/maintenance_logs"
    chunk_size          : int = 500
    chunk_overlap       : int = 50

class AppConfig:
    ollama   = OllamaConfig()
    chroma   = ChromaConfig()
    retriever= RetrieverConfig()
    data     = DataConfig()

cfg = AppConfig()