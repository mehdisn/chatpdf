import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from config import ApplicationConfig
from interface import ChatInterface
from core import DocumentManager, VectorStore, LLMManager
import streamlit as st


logging.basicConfig(level=logging.INFO)

@st.cache_resource
def get_llm_manager(config):
    return LLMManager(config)

@st.cache_resource
def get_cached_vector_store(config):
    return VectorStore.get_cached_vector_store(config)

def main():
    config = ApplicationConfig()
    doc_manager = DocumentManager(config)
    vector_store = get_cached_vector_store(config)
    llm_manager = get_llm_manager(config)
    chat_interface = ChatInterface(doc_manager, vector_store, llm_manager)
    chat_interface.render()

if __name__ == "__main__":
    main()