import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

@dataclass
class ApplicationConfig:
    """Configuration management for the application."""
    UPLOAD_DIR: Path = Path("uploads")
    CHROMA_DIR: Path = Path("chroma_db")
    CACHE_DIR: Path = Path("./models")
    EMBEDDING_MODEL: str = "parsbert/parsbert-base"
    LLM_MODEL: str = "PersianLLaMA/PersianLLaMA-1.1B"
    CHUNK_SIZE: int = 500 
    CHUNK_OVERLAP: int = 50  
    MILVUS_HOST: str = "127.0.0.1"
    MILVUS_PORT: str = 19530

    def __post_init__(self):
        load_dotenv()
        for directory in [self.UPLOAD_DIR, self.CHROMA_DIR, self.CACHE_DIR]:
            directory.mkdir(exist_ok=True)
