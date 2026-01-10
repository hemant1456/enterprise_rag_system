# RAG System Study Plan: Build Your Own Enterprise RAG System

## Prerequisites
- ✅ Understanding of LLMs (Large Language Models)
- ✅ Basic Python knowledge
- ✅ Familiarity with APIs (REST APIs)
- ❌ No prior RAG knowledge required

---

## Table of Contents
1. [Understanding RAG Fundamentals](#1-understanding-rag-fundamentals)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Step-by-Step Implementation Guide](#3-step-by-step-implementation-guide)
4. [Deep Dive into Each Component](#4-deep-dive-into-each-component)
5. [Modification and Extension Guide](#5-modification-and-extension-guide)
6. [Practice Exercises](#6-practice-exercises)

---

## 1. Understanding RAG Fundamentals

### 1.1 What is RAG?
**Retrieval-Augmented Generation (RAG)** combines two key components:
- **Retrieval**: Finding relevant information from a knowledge base
- **Generation**: Using an LLM to generate answers based on retrieved information

### 1.2 Why RAG?
- LLMs have knowledge cutoff dates
- LLMs can hallucinate (make up information)
- RAG provides up-to-date, accurate information from your documents
- RAG allows you to cite sources

### 1.3 RAG Workflow
```
Document → Chunking → Embeddings → Vector Store
                                    ↓
User Question → Embedding → Similarity Search → Retrieve Relevant Chunks
                                                      ↓
                                              LLM + Context → Answer
```

### 1.4 Key Concepts to Learn
1. **Embeddings**: Numerical representations of text that capture semantic meaning
2. **Vector Stores**: Databases optimized for storing and searching embeddings
3. **Similarity Search**: Finding documents similar to a query using cosine similarity or other metrics
4. **Chunking**: Breaking documents into smaller pieces for better retrieval

**Study Time**: 2-3 hours
**Resources**:
- Read about word embeddings (Word2Vec, BERT, sentence transformers)
- Understand cosine similarity
- Learn about vector databases

---

## 2. System Architecture Overview

### 2.1 Component Breakdown

Our system has 6 main components:

1. **Document Processor** (`src/document_processor.py`)
   - Extracts text from DOCX files
   - Handles paragraphs and tables

2. **Text Chunker** (`src/chunker.py`)
   - Splits documents into manageable chunks
   - Maintains context with overlap

3. **Vector Store** (`src/vector_store.py`)
   - Creates embeddings from text
   - Stores embeddings in ChromaDB
   - Performs similarity search

4. **RAG Pipeline** (`src/rag_pipeline.py`)
   - Combines retrieval and generation
   - Uses LangChain for orchestration
   - Integrates with Anthropic Claude

5. **API Layer** (`src/api.py`)
   - FastAPI REST endpoints
   - Handles ingestion and queries

6. **Configuration** (`config.py`)
   - Manages settings and API keys

### 2.2 Data Flow
```
RAG_Enterprise_Dataset.docx
    ↓
Document Processor (extract text)
    ↓
Text Chunker (split into chunks)
    ↓
Vector Store (create embeddings, store in ChromaDB)
    ↓
[User Query]
    ↓
Vector Store (similarity search)
    ↓
RAG Pipeline (retrieve chunks + generate answer)
    ↓
API Response (answer + sources)
```

**Study Time**: 1-2 hours
**Action**: Read through each file and understand its purpose

---

## 3. Step-by-Step Implementation Guide

### Phase 1: Document Processing (Week 1)

#### Step 1.1: Understand Document Extraction
**File**: `src/document_processor.py`

**What to Learn**:
- How `python-docx` library works
- Extracting text from paragraphs
- Extracting text from tables
- Error handling for missing files

**Practice**:
1. Create a simple script that extracts text from a DOCX file
2. Modify the processor to extract only headings
3. Add support for extracting images metadata

**Key Code to Understand**:
```python
from docx import Document
doc = Document('file.docx')
for paragraph in doc.paragraphs:
    print(paragraph.text)
```

**Study Time**: 2-3 hours

---

#### Step 1.2: Implement Text Chunking
**File**: `src/chunker.py`

**What to Learn**:
- Why chunking is necessary (token limits, context windows)
- Different chunking strategies:
  - Fixed-size chunks
  - Sentence-based chunks
  - Semantic chunks
- Overlap strategy (why we need it)

**Practice**:
1. Experiment with different chunk sizes (500, 1000, 2000 characters)
2. Try different overlap values (0, 100, 200 characters)
3. Implement a custom chunker that splits on sentences

**Key Code to Understand**:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_text(text)
```

**Study Time**: 3-4 hours

---

### Phase 2: Vector Embeddings (Week 2)

#### Step 2.1: Understanding Embeddings
**What to Learn**:
- What are embeddings? (vectors that represent meaning)
- How embeddings work (neural networks learn semantic relationships)
- Popular embedding models:
  - OpenAI embeddings
  - HuggingFace sentence transformers
  - Cohere embeddings

**Practice**:
1. Install sentence-transformers
2. Create embeddings for a few sentences
3. Calculate similarity between sentences using cosine similarity

**Key Code to Understand**:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(['Hello world', 'Hi there'])
# embeddings is a numpy array of shape (2, 384)
```

**Study Time**: 4-5 hours

---

#### Step 2.2: Vector Stores
**File**: `src/vector_store.py`

**What to Learn**:
- What is a vector store? (database for embeddings)
- Popular vector databases:
  - ChromaDB (local, easy)
  - Pinecone (cloud, managed)
  - Weaviate (self-hosted)
  - Qdrant (high performance)
- How similarity search works

**Practice**:
1. Create a simple vector store with 10 documents
2. Perform similarity search
3. Experiment with different similarity metrics

**Key Code to Understand**:
```python
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings_model
)
results = vectorstore.similarity_search(query, k=3)
```

**Study Time**: 5-6 hours

---

### Phase 3: RAG Pipeline (Week 3)

#### Step 3.1: Understanding LangChain
**What to Learn**:
- What is LangChain? (framework for LLM applications)
- Key LangChain concepts:
  - Chains (combining multiple steps)
  - Retrievers (getting documents)
  - Prompt templates
- RetrievalQA chain

**Practice**:
1. Create a simple LangChain chain
2. Build a custom prompt template
3. Experiment with different chain types

**Study Time**: 4-5 hours

---

#### Step 3.2: Building the RAG Pipeline
**File**: `src/rag_pipeline.py`

**What to Learn**:
- How retrieval and generation work together
- Prompt engineering for RAG
- Handling context windows
- Source citation

**Practice**:
1. Modify the prompt template to get different answer styles
2. Experiment with different numbers of retrieved chunks (k=1, 3, 5)
3. Add temperature and other LLM parameters

**Key Code to Understand**:
```python
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
result = qa_chain({"query": question})
```

**Study Time**: 6-8 hours

---

### Phase 4: API and Integration (Week 4)

#### Step 4.1: FastAPI Basics
**File**: `src/api.py`

**What to Learn**:
- FastAPI framework
- REST API design
- Request/response models (Pydantic)
- Error handling

**Practice**:
1. Create a simple FastAPI app with one endpoint
2. Add request validation
3. Implement error handling

**Key Code to Understand**:
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query(request: QueryRequest):
    return {"answer": "..."}
```

**Study Time**: 3-4 hours

---

#### Step 4.2: System Integration
**What to Learn**:
- Connecting all components
- State management (global variables vs dependency injection)
- Startup/shutdown events
- Configuration management

**Practice**:
1. Add a new endpoint for listing all documents
2. Implement document deletion
3. Add logging and monitoring

**Study Time**: 4-5 hours

---

## 4. Deep Dive into Each Component

### 4.1 Document Processor Deep Dive

**File**: `src/document_processor.py`

**Key Concepts**:
- File I/O and error handling
- Document structure (paragraphs, tables, headers)
- Text extraction strategies

**Modification Ideas**:
1. Add support for PDF files (use PyPDF2 or pdfplumber)
2. Extract metadata (author, creation date)
3. Handle images and extract alt text
4. Support multiple file formats

**Code Walkthrough**:
```python
# Line by line explanation:
doc = Document(self.file_path)  # Open DOCX file
for paragraph in doc.paragraphs:  # Iterate paragraphs
    if paragraph.text.strip():  # Skip empty paragraphs
        text_parts.append(paragraph.text.strip())  # Collect text
```

---

### 4.2 Chunker Deep Dive

**File**: `src/chunker.py`

**Key Concepts**:
- RecursiveCharacterTextSplitter algorithm
- Separators hierarchy (paragraphs → sentences → words)
- Overlap importance (maintains context)

**Modification Ideas**:
1. Implement semantic chunking (group related sentences)
2. Add chunk metadata (section, page number)
3. Implement sliding window chunking
4. Create chunker that respects document structure

**Code Walkthrough**:
```python
# The splitter tries separators in order:
separators=["\n\n", "\n", ". ", " ", ""]
# First tries paragraphs, then lines, then sentences, etc.
```

---

### 4.3 Vector Store Deep Dive

**File**: `src/vector_store.py`

**Key Concepts**:
- Embedding model initialization
- ChromaDB persistence
- Similarity search algorithms
- Metadata filtering

**Modification Ideas**:
1. Add support for multiple embedding models
2. Implement hybrid search (keyword + semantic)
3. Add metadata filtering (filter by date, author)
4. Implement re-ranking of results

**Code Walkthrough**:
```python
# Creating embeddings and storing:
self.vectorstore = Chroma.from_texts(
    texts=texts,  # Your chunks
    embedding=self.embeddings,  # Embedding model
    persist_directory="./chroma_db",  # Where to save
    metadatas=metadatas  # Optional metadata
)

# Searching:
results = self.vectorstore.similarity_search_with_score(query, k=3)
# Returns top 3 most similar documents with similarity scores
```

---

### 4.4 RAG Pipeline Deep Dive

**File**: `src/rag_pipeline.py`

**Key Concepts**:
- RetrievalQA chain architecture
- Prompt engineering
- Context window management
- Source document tracking

**Modification Ideas**:
1. Implement different chain types:
   - "map_reduce" (for long documents)
   - "refine" (iterative refinement)
2. Add re-ranking of retrieved chunks
3. Implement query expansion
4. Add conversation memory

**Code Walkthrough**:
```python
# The chain does this:
# 1. Takes user query
# 2. Retrieves relevant chunks (via retriever)
# 3. Combines chunks into context
# 4. Sends context + query to LLM
# 5. Returns answer + sources

result = self.qa_chain({"query": question})
# Returns: {"result": "answer", "source_documents": [...]}
```

---

### 4.5 API Deep Dive

**File**: `src/api.py`

**Key Concepts**:
- FastAPI routing
- Async/await patterns
- Global state management
- Error handling and logging

**Modification Ideas**:
1. Add authentication (API keys, JWT tokens)
2. Implement rate limiting
3. Add request logging
4. Create admin endpoints (delete documents, view stats)
5. Add WebSocket support for streaming responses

**Code Walkthrough**:
```python
# Global state (shared across requests):
vector_store = None
rag_pipeline = None

# On startup, initialize vector store
@app.on_event("startup")
async def startup_event():
    global vector_store
    vector_store = VectorStore(...)

# Endpoint handles request:
@app.post("/query")
async def query_document(request: QueryRequest):
    # Use global rag_pipeline
    result = rag_pipeline.query(request.question)
    return result
```

---

## 5. Modification and Extension Guide

### 5.1 Adding New Document Types

**Current**: Only DOCX files
**Goal**: Support PDF, TXT, MD files

**Steps**:
1. Install required libraries (PyPDF2, markdown)
2. Create a document processor factory
3. Add file type detection
4. Implement processors for each type

**Example**:
```python
class DocumentProcessorFactory:
    @staticmethod
    def create_processor(file_path):
        if file_path.endswith('.docx'):
            return DocxProcessor(file_path)
        elif file_path.endswith('.pdf'):
            return PdfProcessor(file_path)
        elif file_path.endswith('.txt'):
            return TxtProcessor(file_path)
```

---

### 5.2 Changing Embedding Models

**Current**: HuggingFace sentence-transformers
**Goal**: Use OpenAI embeddings or Cohere

**Steps**:
1. Update `requirements.txt`
2. Modify `vector_store.py` initialization
3. Update configuration

**Example**:
```python
# For OpenAI:
from langchain_openai import OpenAIEmbeddings
self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# For Cohere:
from langchain_community.embeddings import CohereEmbeddings
self.embeddings = CohereEmbeddings(cohere_api_key=api_key)
```

---

### 5.3 Adding Multiple Documents

**Current**: Single document ingestion
**Goal**: Support multiple documents, document management

**Steps**:
1. Add document metadata (name, upload date, etc.)
2. Create document collection management
3. Add endpoint to list/delete documents
4. Implement document-specific retrieval

**Example**:
```python
# Store document info:
documents = {
    "doc1": {"name": "manual.pdf", "chunks": [0, 1, 2]},
    "doc2": {"name": "guide.docx", "chunks": [3, 4, 5]}
}

# Filter by document:
results = vectorstore.similarity_search(
    query,
    filter={"source": "manual.pdf"}
)
```

---

### 5.4 Improving Retrieval Quality

**Current**: Simple similarity search
**Goal**: Better retrieval with re-ranking, query expansion

**Steps**:
1. Implement BM25 for keyword search
2. Add hybrid search (combine semantic + keyword)
3. Implement re-ranking with cross-encoders
4. Add query expansion (generate related queries)

**Example**:
```python
# Hybrid search:
semantic_results = vectorstore.similarity_search(query, k=10)
keyword_results = bm25_search(query, k=10)
combined = merge_and_rerank(semantic_results, keyword_results)
```

---

### 5.5 Adding Conversation Memory

**Current**: Stateless (each query is independent)
**Goal**: Remember conversation history

**Steps**:
1. Add conversation session management
2. Store conversation history
3. Include history in prompt
4. Implement context window management

**Example**:
```python
conversations = {
    "session_123": [
        {"role": "user", "content": "What is X?"},
        {"role": "assistant", "content": "X is..."}
    ]
}

# Include in prompt:
prompt = f"""
Previous conversation:
{format_history(conversations[session_id])}

Current question: {question}
"""
```

---

### 5.6 Adding Authentication

**Current**: No authentication
**Goal**: Secure API with user authentication

**Steps**:
1. Install FastAPI security dependencies
2. Implement JWT token generation
3. Add authentication middleware
4. Create user management

**Example**:
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/query")
async def query(
    request: QueryRequest,
    token: str = Depends(security)
):
    # Verify token
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401)
    # Process query...
```

---

## 6. Practice Exercises

### Exercise 1: Basic RAG (Week 1)
**Goal**: Build a minimal RAG system from scratch

**Requirements**:
- Extract text from a single document
- Create 5 chunks
- Generate embeddings (use a simple method)
- Store in a list (no vector DB needed)
- Implement simple similarity search
- Generate answer with LLM

**Expected Time**: 4-6 hours

---

### Exercise 2: Improve Chunking (Week 2)
**Goal**: Implement better chunking strategies

**Requirements**:
- Implement sentence-aware chunking
- Add chunk overlap
- Preserve document structure (headings, sections)
- Test with different chunk sizes

**Expected Time**: 3-4 hours

---

### Exercise 3: Multiple Documents (Week 3)
**Goal**: Support multiple documents

**Requirements**:
- Allow uploading multiple documents
- Track which chunk belongs to which document
- Filter results by document
- Add document management endpoints

**Expected Time**: 5-6 hours

---

### Exercise 4: Advanced Retrieval (Week 4)
**Goal**: Improve retrieval quality

**Requirements**:
- Implement BM25 keyword search
- Combine semantic + keyword search
- Add result re-ranking
- Compare retrieval quality

**Expected Time**: 6-8 hours

---

### Exercise 5: Production Features (Week 5)
**Goal**: Add production-ready features

**Requirements**:
- Add authentication
- Implement rate limiting
- Add logging and monitoring
- Create unit tests
- Add error recovery

**Expected Time**: 8-10 hours

---

## 7. Learning Resources

### Books
- "Building LLM Applications for Production" by LlamaIndex team
- "LangChain in Action" by Harrison Chase

### Online Courses
- LangChain official documentation and tutorials
- Vector databases course (Pinecone, ChromaDB docs)
- FastAPI official tutorial

### Documentation
- [LangChain Docs](https://python.langchain.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Anthropic Claude API Docs](https://docs.anthropic.com/)

### YouTube Channels
- LangChain YouTube channel
- Pinecone vector database tutorials
- FastAPI tutorials

---

## 8. Study Schedule

### Week 1: Foundations
- Day 1-2: Understand RAG concepts
- Day 3-4: Document processing and chunking
- Day 5-7: Practice Exercise 1

### Week 2: Embeddings and Vector Stores
- Day 1-2: Understanding embeddings
- Day 3-4: Vector stores and similarity search
- Day 5-7: Practice Exercise 2

### Week 3: RAG Pipeline
- Day 1-2: LangChain basics
- Day 3-4: Building RAG pipeline
- Day 5-7: Practice Exercise 3

### Week 4: API and Integration
- Day 1-2: FastAPI basics
- Day 3-4: System integration
- Day 5-7: Practice Exercise 4

### Week 5: Advanced Topics
- Day 1-2: Advanced retrieval
- Day 3-4: Production features
- Day 5-7: Practice Exercise 5

---

## 9. Troubleshooting Guide

### Common Issues

**Issue**: Embeddings not working
- **Solution**: Check if sentence-transformers is installed in correct environment
- **Check**: `python -c "import sentence_transformers"`

**Issue**: Vector store errors
- **Solution**: Delete `chroma_db` folder and re-ingest
- **Check**: Permissions on chroma_db directory

**Issue**: LLM API errors
- **Solution**: Verify API key in .env file
- **Check**: Model name is correct and available

**Issue**: Chunking too small/large
- **Solution**: Adjust `chunk_size` and `chunk_overlap` in config
- **Experiment**: Try values between 500-2000 characters

**Issue**: Poor retrieval quality
- **Solution**: Increase `top_k` value, try different embedding models
- **Check**: Chunk size might be too large/small

---

## 10. Next Steps After Mastery

Once you understand the system:

1. **Optimize Performance**
   - Implement caching
   - Use async processing
   - Optimize embedding generation

2. **Scale the System**
   - Add load balancing
   - Implement distributed vector stores
   - Use message queues

3. **Advanced RAG Techniques**
   - Implement query routing
   - Add multi-step reasoning
   - Implement self-correction

4. **Monitoring and Observability**
   - Add metrics collection
   - Implement tracing
   - Create dashboards

5. **Build Your Own RAG Product**
   - Identify use case
   - Design architecture
   - Implement and deploy

---

## Conclusion

This study plan provides a comprehensive path to understanding and building RAG systems. Follow it step-by-step, practice with exercises, and don't hesitate to experiment and modify the code.

**Remember**: The best way to learn is by doing. Modify the code, break things, fix them, and understand why they work.

Good luck on your RAG journey! 🚀
