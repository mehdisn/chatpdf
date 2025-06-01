# ChatPDF - Persian RAG

An application that allows users to chat with their PDF documents in Persian (Farsi). This tool uses advanced language models to understand and respond to questions about the content of a PDF file.

## Features

- 📄 Upload and process PDF documents
- 💬 Chat interface for interacting with document content
- 🗣️ Persian language support with specialized models
- 🔍 Advanced document search and retrieval
- 🧠 Powered by ParsBERT and PersianLLaMA models
- 🗄️ Milvus vector database for efficient similarity search
- 🎨 Streamlit-based user interface

## Prerequisites

- Python 3.8 or higher
- Docker (for running Milvus)
- pip (Python package manager)

## Configuration

The application uses the following models by default:
- Embedding Model: `parsbert/parsbert-base`
- LLM Model: `PersianLLaMA/PersianLLaMA-1.1B`

You can modify these settings in `src/config/setting.py`.

### Milvus Configuration
The application uses Milvus as its vector database for efficient similarity search. By default, it connects to:
- Host: 127.0.0.1
- Port: 19530

To run Milvus locally using Docker:
```bash
docker run -d --name milvus_standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
```

### Streamlit Interface
The application uses Streamlit for its web interface. The interface provides:
- PDF file upload functionality
- Chat interface for document interaction
- Real-time response streaming

## Project Structure

```
chatpdf/
├── src/
│   ├── config/         # Configuration files
│   ├── core/          # Core functionality
│   ├── interface/     # User interface components
│   └── main.py        # Application entry point
├── models/            # Downloaded model files
├── uploads/           # Uploaded PDF files
└── README.md
```

## Usage

1. Start Milvus (if not already running):
```bash
docker start milvus_standalone
```

2. Start the application:
```bash
streamlit run src/main.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## TODO

### Interface Improvements
- [ ] Add loading indicators for PDF processing
- [ ] Implement better error handling and user feedback
- [ ] Add support for multiple document uploads
- [ ] Add chat history display and management
- [ ] Add document preview functionality

### Technical Improvements
- [ ] Optimize Milvus query performance
- [ ] Implement caching for frequently accessed documents
- [ ] Add support for different PDF formats and encodings
- [ ] Implement proper session management
- [ ] Add user authentication system
- [ ] Implement proper logging system
- [ ] Add unit tests and integration tests
- [ ] Optimize model loading and inference time

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
