"""RAG pipeline module combining retrieval and generation."""
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from typing import Optional
import logging

from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Combines retrieval and generation for question answering."""
    
    def __init__(self, vector_store: VectorStore, anthropic_api_key: str, top_k: int = 3):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Initialized vector store instance
            anthropic_api_key: Anthropic API key
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.top_k = top_k
        
        # Initialize Anthropic LLM
        self.llm = ChatAnthropic(
            anthropic_api_key=anthropic_api_key,
            model_name="claude-3-sonnet-20240229"  # Default Claude model
        )
        
        # Create custom prompt template
        self.prompt_template = PromptTemplate(
            template="""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so.

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Initialize the QA chain
        self.qa_chain: Optional[RetrievalQA] = None
        self._initialize_chain()
    
    def _initialize_chain(self):
        """Initialize the retrieval QA chain."""
        try:
            retriever = self.vector_store.get_retriever(k=self.top_k)
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt_template}
            )
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {str(e)}")
            raise
    
    def query(self, question: str) -> dict:
        """
        Query the RAG pipeline with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("RAG pipeline not initialized")
        
        try:
            result = self.qa_chain({"query": question})
            
            return {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
        except Exception as e:
            logger.error(f"Error querying RAG pipeline: {str(e)}")
            raise
