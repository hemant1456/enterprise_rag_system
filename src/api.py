"""FastAPI application for the RAG system."""
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import logging
import os

from config import settings
from src.document_processor import DocumentProcessor
from src.chunker import TextChunker
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise RAG System",
    description="RAG system for querying documents using Anthropic Claude",
    version="1.0.0"
)

# Global instances
vector_store: Optional[VectorStore] = None
rag_pipeline: Optional[RAGPipeline] = None


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    sources: list


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""
    message: str
    chunks_created: int


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global vector_store
    
    try:
        # Initialize vector store (will load if exists)
        vector_store = VectorStore(
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
            embedding_model=settings.embedding_model
        )
        
        # Try to load existing collection
        try:
            vector_store.load_collection()
            logger.info("Loaded existing vector store")
        except Exception:
            logger.info("No existing vector store found, will create on first ingest")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_store_initialized": vector_store is not None,
        "rag_pipeline_initialized": rag_pipeline is not None
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document():
    """
    Process data.docx and create vector embeddings.
    
    This endpoint:
    1. Extracts text from data.docx
    2. Chunks the text
    3. Creates embeddings and stores them in Chroma
    4. Initializes the RAG pipeline
    """
    global vector_store, rag_pipeline
    
    try:
        # Check if data.docx exists
        doc_path = "data.docx"
        if not os.path.exists(doc_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {doc_path}"
            )
        
        # Process document
        logger.info("Starting document ingestion...")
        processor = DocumentProcessor(doc_path)
        text = processor.extract_text()
        
        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document is empty or contains no extractable text"
            )
        
        # Chunk text
        chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        chunks = chunker.chunk_text(text)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create text chunks"
            )
        
        # Create or update vector store
        if vector_store is None:
            vector_store = VectorStore(
                persist_directory=settings.chroma_persist_directory,
                collection_name=settings.chroma_collection_name,
                embedding_model=settings.embedding_model
            )
        
        # Create metadata for each chunk
        metadatas = [{"chunk_index": i, "source": "data.docx"} for i in range(len(chunks))]
        
        # Create collection with chunks
        vector_store.create_collection(chunks, metadatas)
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            anthropic_api_key=settings.anthropic_api_key,
            top_k=settings.top_k
        )
        
        logger.info(f"Successfully ingested document with {len(chunks)} chunks")
        
        return IngestResponse(
            message="Document ingested successfully",
            chunks_created=len(chunks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query the document using RAG.
    
    This endpoint:
    1. Retrieves relevant chunks from the vector store
    2. Generates an answer using Anthropic Claude
    3. Returns the answer with source citations
    """
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document not ingested yet. Please call /ingest first."
        )
    
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )
    
    try:
        logger.info(f"Processing query: {request.question}")
        result = rag_pipeline.query(request.question)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
