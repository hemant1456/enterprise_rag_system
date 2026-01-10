# Enterprise RAG System

A production-ready Retrieval-Augmented Generation (RAG) system that processes DOCX documents and provides a REST API for querying document content using Anthropic Claude.

## Features

- **Document Processing**: Extracts text from DOCX files including paragraphs and tables
- **Intelligent Chunking**: Splits documents into semantically meaningful chunks with overlap
- **Vector Storage**: Uses ChromaDB for efficient similarity search
- **RAG Pipeline**: Combines retrieval and generation using Anthropic Claude
- **REST API**: FastAPI-based API for document ingestion and querying
- **Source Citations**: Returns answers with source document references

## Architecture

```
data.docx → Document Processor → Text Chunker → Embeddings → Chroma DB
                                                                    ↓
API Request → FastAPI → RAG Pipeline ← Retrieval ← Chroma DB
                                    ↓
                              Anthropic Claude
                                    ↓
                              Answer + Sources
```

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```
   
   Optional configuration (with defaults):
   ```
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   TOP_K=3
   API_HOST=0.0.0.0
   API_PORT=8000
   ```

## Usage

### Starting the Server

Run the application:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

Returns the health status of the system.

**Response:**
```json
{
  "status": "healthy",
  "vector_store_initialized": true,
  "rag_pipeline_initialized": true
}
```

#### 2. Ingest Document
```bash
POST /ingest
```

Processes `data.docx` and creates vector embeddings.

**Response:**
```json
{
  "message": "Document ingested successfully",
  "chunks_created": 42
}
```

#### 3. Query Document
```bash
POST /query
Content-Type: application/json

{
  "question": "What is the main topic of the document?"
}
```

**Response:**
```json
{
  "answer": "The document discusses...",
  "sources": [
    {
      "content": "Relevant text excerpt...",
      "metadata": {
        "chunk_index": 0,
        "source": "data.docx"
      }
    }
  ]
}
```

### Example Usage with cURL

```bash
# Ingest the document
curl -X POST http://localhost:8000/ingest

# Query the document
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key points discussed?"}'
```

### Example Usage with Python

```python
import requests

# Ingest document
response = requests.post("http://localhost:8000/ingest")
print(response.json())

# Query document
response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What is this document about?"}
)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} chunks found")
```

## Project Structure

```
enterprise_rag_system/
├── src/
│   ├── __init__.py
│   ├── document_processor.py  # DOCX text extraction
│   ├── chunker.py              # Text chunking logic
│   ├── vector_store.py         # ChromaDB integration
│   ├── rag_pipeline.py         # RAG chain implementation
│   └── api.py                  # FastAPI application
├── config.py                   # Configuration management
├── requirements.txt            # Python dependencies
├── main.py                     # Application entry point
├── data.docx                   # Input document
└── README.md                   # This file
```

## Configuration

The system uses environment variables for configuration. Key settings:

- `ANTHROPIC_API_KEY`: Required - Your Anthropic API key
- `EMBEDDING_MODEL`: Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)
- `CHUNK_SIZE`: Character size for text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K`: Number of documents to retrieve (default: 3)
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)

## Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **LangChain**: Framework for building LLM applications
- **Anthropic Claude**: Large language model for generation
- **ChromaDB**: Vector database for embeddings
- **python-docx**: DOCX file processing
- **HuggingFace Transformers**: Embedding models

## Development

### Running in Development Mode

The server runs with auto-reload enabled by default:
```bash
python main.py
```

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Notes

- The vector database is persisted in the `chroma_db/` directory
- First-time embedding model download may take a few minutes
- Ensure `data.docx` exists in the project root before calling `/ingest`
- The system requires an Anthropic API key for querying

## License

See LICENSE file for details.
