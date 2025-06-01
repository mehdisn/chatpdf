from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
import streamlit as st
import logging

class LoggingHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        print("Chat model started")

    def on_llm_start(self, *args, **kwargs):
        try:
            if st.runtime.exists():
                with st.spinner("Processing..."):
                    pass
        except:
            pass
        self.logger.info("Starting LLM")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        try:
            if st.runtime.exists():
                st.empty()
        except:
            pass
        self.logger.info("LLM finished")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        chain_name = (
            serialized.get("name", "unknown") if isinstance(serialized, dict) else "unknown"
        )
        input_data = (
            inputs.get("question", "No input") if isinstance(inputs, dict) else str(inputs)
        )
        print(f"Chain '{chain_name}' started with input: {input_data}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print(f"Chain ended, outputs: {outputs}")

    def on_llm_error(self, error, **kwargs):
        self.logger.error(f"LLM error: {str(error)}")