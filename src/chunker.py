"""Text chunking module for splitting documents into manageable pieces."""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits text into semantically meaningful chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        chunks = self.splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
