"""Configuration management for the RAG system."""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Anthropic API Configuration
    anthropic_api_key: str
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Default HuggingFace model
    
    # Chroma Configuration
    chroma_persist_directory: str = "./chroma_db"
    chroma_collection_name: str = "enterprise_rag"
    
    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval Configuration
    top_k: int = 3
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
