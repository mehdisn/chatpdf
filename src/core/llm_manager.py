import torch
import logging
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import ApplicationConfig
from .callback_handler import LoggingHandler

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages the language model and inference pipeline."""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self._initialize_model()
        self._setup_prompt_template()
        self.callbacks = LoggingHandler()

    def _initialize_model(self):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.LLM_MODEL,
                cache_dir=self.config.CACHE_DIR,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                max_memory={0: "4GiB", 'cpu': "8GiB"},
                offload_folder="offload"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.LLM_MODEL,
                cache_dir=self.config.CACHE_DIR,
                model_max_length=512,
                padding_side="left",
                truncation=True,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                temperature=0.3,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                return_full_text=False,
                top_k=50,
                top_p=0.9,
                no_repeat_ngram_size=2
            )
            
            self.pipeline = HuggingFacePipeline(
                pipeline=pipe,
                model_kwargs={
                    "temperature": 0.3,
                    "max_length": 512,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1
                }
            )
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise

    def _setup_prompt_template(self):
        self.prompt_template = PromptTemplate(
            template="""Write a response that appropriately completes the request.
### Instruction: {question}
### Input: {context}
### Response: Let me analyze the provided context and answer your question in detail.
""",
            input_variables=["context", "question"]
        )

    def test_llm(self):
        """Test the LLM with a simple greeting message."""
        try:
            test_prompt = PromptTemplate(
                template="""Below is an instruction in English language. Write a response in English that appropriately completes the request.

### Instruction: Hello, please introduce yourself.

### Response: Let me introduce myself.
""",
                input_variables=[]
            )

            test_chain = test_prompt | self.pipeline | StrOutputParser()
            test_chain_with_callbacks = test_chain.with_config(callbacks=[self.callbacks])
            
            return test_chain_with_callbacks.invoke({})
        except Exception as e:
            logger.error(f"Error testing LLM: {str(e)}")
            raise

    def create_qa_chain(self, retriever):
        def format_docs(docs):
            formatted_docs = "\n\n---\n\n".join(doc.page_content for doc in docs[:3])
            return formatted_docs[:1500]

        try:
            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | self.prompt_template
                | self.pipeline
                | StrOutputParser()
            )
            
            chain_with_callbacks = rag_chain.with_config(
                callbacks=[self.callbacks],
                config={
                    "timeout": 60,
                    "max_retries": 2,
                    "retry_on_timeout": True,
                    "max_concurrent_requests": 1
                }
            )
            return chain_with_callbacks
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise
