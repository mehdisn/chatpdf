from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
from src.config import ApplicationConfig
logger = logging.getLogger(__name__)


class DocumentManager:
    """Handles document processing and chunking operations."""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

    def process_pdf(self, pdf_path: str) -> Optional[List[Document]]:
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            return self.text_splitter.split_documents(pages)
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
