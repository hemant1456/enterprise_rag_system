"""Vector store module for managing embeddings in Chroma."""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Optional
import logging
import os

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector embeddings in Chroma database."""
    
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist Chroma database
            collection_name: Name of the Chroma collection
            embedding_model: Embedding model name (optional)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embeddings
        # Use HuggingFace embeddings (free, no API key required)
        # Default model: sentence-transformers/all-MiniLM-L6-v2
        model_name = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'}
            )
            logger.info(f"Initialized embeddings with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise ValueError(f"Failed to initialize embeddings with model {model_name}: {str(e)}")
        
        self.vectorstore: Optional[Chroma] = None
    
    def create_collection(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """
        Create a new collection with texts and their embeddings.
        
        Args:
            texts: List of text chunks to embed and store
            metadatas: Optional list of metadata dictionaries for each text
        """
        if not texts:
            logger.warning("No texts provided for vector store")
            return
        
        try:
            # Use embedding parameter for LangChain Chroma
            if self.embeddings is None:
                raise ValueError("Embeddings not initialized")
            self.vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                metadatas=metadatas or [{}] * len(texts)
            )
            
            logger.info(f"Created vector store with {len(texts)} documents")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_collection(self):
        """Load an existing collection from disk."""
        try:
            if self.embeddings is None:
                raise ValueError("Embeddings not initialized")
            # Use embedding_function for loading existing collections
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            logger.info("Loaded existing vector store")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 3) -> List[dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of dictionaries containing document content and metadata
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_collection or load_collection first.")
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in results
            ]
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def get_retriever(self, k: int = 3):
        """
        Get a LangChain retriever for use in RAG pipeline.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            LangChain retriever
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_collection or load_collection first.")
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
