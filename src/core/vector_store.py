from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from typing import List
import logging
from src.config import ApplicationConfig
from langchain_core.documents import Document
from streamlit import cache_resource

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages document embeddings and similarity search using Milvus."""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self._initialize_embeddings()
        self.vector_store = None

    def _initialize_embeddings(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                cache_folder=str(self.config.CACHE_DIR),
                model_kwargs={"device": "cpu"}
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]) -> None:
        try:
            print("LOAD Milvus add_documents")
            self.vector_store = Milvus.from_documents(
                documents=documents,
                embedding=self.embeddings,
                drop_old=True,
                connection_args={
                    "host": self.config.MILVUS_HOST,
                    "port": self.config.MILVUS_PORT
                }
            )
            print("LOAD Milvus add_documents END")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def get_retriever(self):
        if self.vector_store is None:
            raise ValueError("Vector store has not been initialized with documents")
        return self.vector_store.as_retriever(
            search_kwargs={"k": 3}
        )

    @staticmethod
    @cache_resource
    def get_cached_vector_store(config, _documents=None):
        vector_store = VectorStore(config)
        if _documents:
            vector_store.add_documents(_documents)
        return vector_store