import streamlit as st
from pathlib import Path
import tempfile
import logging
from src.core import DocumentManager, VectorStore, LLMManager
from contextlib import contextmanager
import concurrent.futures
import threading

logger = logging.getLogger(__name__)

@contextmanager
def streamlit_context():
    try:
        if not st.runtime.exists():
            import streamlit.runtime.scriptrunner.script_runner as script_runner
            script_runner.add_script_run_ctx()
        yield
    finally:
        pass

class ChatInterface:
    """Manages the Streamlit user interface with Persian support."""

    def __init__(
            self,
            doc_manager: DocumentManager,
            vector_store: VectorStore,
            llm_manager: LLMManager
    ):
        self.doc_manager = doc_manager
        self.vector_store = vector_store
        self.llm_manager = llm_manager
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = None
        if 'retriever' not in st.session_state:
            st.session_state.retriever = None

    def render(self):
        """Render the Streamlit interface."""
        st.title("ChatPDF.ir - چت با پی‌دی‌اف")

        # Add a test button
        if st.button("تست مدل هوش مصنوعی"):
            with st.spinner("در حال تست مدل..."):
                try:
                    response = self.llm_manager.test_llm()
                    st.success("مدل با موفقیت تست شد!")
                    st.markdown("### پاسخ مدل:")
                    st.write(response)
                except Exception as e:
                    st.error(f"خطا در تست مدل: {str(e)}")

        st.markdown("""
        فایل PDF خود را آپلود کنید و سوالات خود را در مورد محتوای آن بپرسید.
        سیستم هوش مصنوعی ما اطلاعات مرتبط را از سند پیدا خواهد کرد.
        """)

        self._handle_file_upload()
        self._render_chat_interface()

    def _handle_file_upload(self):
        """Handle PDF file upload and processing."""
        uploaded_file = st.file_uploader(
            "PDF خود را آپلود کنید",
            type="pdf",
            help="برای شروع پرسش و پاسخ، یک فایل PDF آپلود کنید"
        )

        if uploaded_file:
            self._process_uploaded_file(uploaded_file)

    def _process_uploaded_file(self, uploaded_file) -> None:
        """Process the uploaded PDF file and initialize the QA chain."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            with st.spinner("در حال پردازش PDF..."):
                docs = self.doc_manager.process_pdf(tmp_path)
                if docs:
                    # Use the cached vector store
                    cached_vector_store = VectorStore.get_cached_vector_store(
                        self.llm_manager.config, _documents=docs
                    )
                    logger.info("Vector store created successfully.")
                    retriever = cached_vector_store.get_retriever()
                    st.session_state.retriever = retriever  # Store the retriever in session state
                    st.session_state.qa_chain = self.llm_manager.create_qa_chain(retriever)
                    st.session_state.pdf_processed = True
                    st.success("PDF با موفقیت پردازش شد!")
                else:
                    st.error("استخراج متن از PDF امکان‌پذیر نبود.")

            Path(tmp_path).unlink()

        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            st.error("خطایی در پردازش PDF رخ داد.")

    def _render_chat_interface(self):
        """Render the chat interface and handle user queries."""
        if st.session_state.pdf_processed:
            query = st.text_input(
                "سوال خود را در مورد PDF بپرسید:",
                key="query_input"
            )

            if query:
                self._handle_query(query)

    def _handle_query(self, query: str):
        """Handle user query and display the response."""
        with streamlit_context():
            try:
                response_container = st.empty()
                with st.spinner("در حال جستجوی اطلاعات مرتبط..."):
                    qa_chain = st.session_state.qa_chain
                    logger.info(f"Query: {query}")
                    logger.info(f"QA Chain Type: {type(qa_chain)}")

                    try:
                        result = run_with_timeout(
                            qa_chain.invoke,
                            (query,),
                            timeout=360
                        )
                    except TimeoutError:
                        st.error("پاسخ‌گویی به دلیل محدودیت زمانی با مشکل مواجه شد. لطفاً دوباره تلاش کنید.")
                        return
                    except Exception as e:
                        logger.error(f"Error during query execution: {str(e)}")
                        st.error("خطایی در پردازش سوال رخ داد.")
                        return

                    logger.info(f"Result Type: {type(result)}")
                    logger.info(f"Result: {result}")

                    if isinstance(result, dict):
                        response = result.get("result", "No response found.")
                        context = result.get("context", "No context found.")
                    else:
                        response = result
                        context = "No context found."

                    st.markdown("### پاسخ:")
                    st.write(response)

                    st.markdown("### زمینه:")
                    st.write(context)

                    st.markdown("### منابع مرتبط:")
                    retriever = st.session_state.retriever
                    if retriever:
                        try:
                            source_documents = run_with_timeout(
                                retriever.get_relevant_documents,
                                (query,),
                                timeout=110
                            )
                            for i, doc in enumerate(source_documents, 1):
                                st.markdown(f"""
                                **بخش {i}**
                                > {doc.page_content}
                                """)

                            st.session_state.chat_history.append({
                                "query": query,
                                "response": response,
                                "sources": source_documents
                            })
                        except TimeoutError:
                            st.warning("زمان جستجوی منابع به پایان رسید.")
                        except Exception as e:
                            logger.error(f"Error retrieving documents: {str(e)}")
                            st.warning("خطا در بازیابی منابع مرتبط.")
                    else:
                        st.warning("هیچ منبع مرتبطی یافت نشد.")

            except Exception as e:
                logger.error(f"Error handling query: {str(e)}", exc_info=True)
                st.error("خطایی در پردازش سوال شما رخ داد.")

def run_with_timeout(func, args, timeout):
    """Run a function with timeout that works on all platforms."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=360)
        except concurrent.futures.TimeoutError:
            executor._threads.clear()
            concurrent.futures.thread._threads_queues.clear()
            raise TimeoutError("Operation timed out")