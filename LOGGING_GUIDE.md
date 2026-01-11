# Complete Logging Guide for Enterprise RAG System

## Table of Contents
1. [What is Logging and Why Do We Need It?](#1-what-is-logging-and-why-do-we-need-it)
2. [Python Logging Module Overview](#2-python-logging-module-overview)
3. [Current Logging in Our Project](#3-current-logging-in-our-project)
4. [Understanding Log Levels](#4-understanding-log-levels)
5. [Logging Components Explained](#5-logging-components-explained)
6. [How Logging Works in Our Code](#6-how-logging-works-in-our-code)
7. [Advanced Logging Techniques](#7-advanced-logging-techniques)
8. [Best Practices for RAG Systems](#8-best-practices-for-rag-systems)
9. [Improving Logging in Our Project](#9-improving-logging-in-our-project)
10. [Practical Examples](#10-practical-examples)

---

## 1. What is Logging and Why Do We Need It?

### 1.1 What is Logging?
**Logging** is the process of recording events, messages, and information about your application's execution. Think of it as a diary that your application writes to, documenting what it's doing, when it's doing it, and if anything goes wrong.

### 1.2 Why is Logging Important?

**For Debugging**:
- When something breaks, logs tell you what happened
- You can trace the execution flow
- Identify where errors occurred

**For Monitoring**:
- Track system performance
- Monitor user activity
- Detect issues before they become critical

**For Production**:
- Understand how your system behaves in real-world conditions
- Debug issues without access to the code
- Track API usage and patterns

**For RAG Systems Specifically**:
- Track document ingestion success/failure
- Monitor query performance
- Debug retrieval quality issues
- Track embedding generation time
- Monitor LLM API calls and costs

### 1.3 Logging vs Print Statements

**❌ Don't use print():**
```python
print("Processing document...")  # Bad!
print(f"Error: {error}")  # Bad!
```

**✅ Use logging:**
```python
logger.info("Processing document...")  # Good!
logger.error(f"Error: {error}")  # Good!
```

**Why?**
- Logs can be filtered by level (only show errors, or only show info)
- Logs can be sent to files, databases, or monitoring systems
- Logs include timestamps, module names, and log levels automatically
- Print statements clutter your code and can't be easily controlled

---

## 2. Python Logging Module Overview

### 2.1 Basic Components

Python's logging module has 4 main components:

1. **Logger**: The object you use to log messages
2. **Handler**: Where logs go (console, file, etc.)
3. **Formatter**: How logs are displayed
4. **Filter**: What logs to include/exclude

```
Logger → Handler → Formatter → Output (console/file/etc.)
```

### 2.2 Quick Start

**Minimal Example:**
```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create a logger
logger = logging.getLogger(__name__)

# Use it
logger.info("This is an info message")
logger.error("This is an error message")
```

**Output:**
```
INFO:__main__:This is an info message
ERROR:__main__:This is an error message
```

---

## 3. Current Logging in Our Project

### 3.1 How We Configure Logging

**File**: `src/api.py` (lines 15-19)

```python
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
```

**What this does:**
- Sets log level to INFO (shows INFO, WARNING, ERROR, CRITICAL)
- Formats logs with: timestamp, module name, level, message
- Creates a logger for the current module

### 3.2 Logging Usage Across Our Files

**Document Processor** (`src/document_processor.py`):
```python
logger.info(f"Successfully extracted {len(full_text)} characters")
logger.warning(f"No text content found in {self.file_path}")
logger.error(f"Error processing document: {str(e)}")
```

**Vector Store** (`src/vector_store.py`):
```python
logger.info(f"Initialized embeddings with model: {model_name}")
logger.info(f"Created vector store with {len(texts)} documents")
logger.error(f"Error creating vector store: {str(e)}")
```

**RAG Pipeline** (`src/rag_pipeline.py`):
```python
logger.info("RAG pipeline initialized successfully")
logger.error(f"Error querying RAG pipeline: {str(e)}")
```

**API** (`src/api.py`):
```python
logger.info("Starting document ingestion...")
logger.info(f"Processing query: {request.question}")
logger.error(f"Error during ingestion: {str(e)}")
```

---

## 4. Understanding Log Levels

### 4.1 Log Level Hierarchy

Log levels are ordered by severity (lowest to highest):

```
DEBUG < INFO < WARNING < ERROR < CRITICAL
```

**When you set a level, you see that level and all higher levels.**

Example: If you set level to `INFO`, you'll see:
- ✅ INFO
- ✅ WARNING
- ✅ ERROR
- ✅ CRITICAL
- ❌ DEBUG (filtered out)

### 4.2 When to Use Each Level

#### DEBUG
**Use for**: Detailed information for debugging
**Example**:
```python
logger.debug(f"Processing chunk {i} of {total_chunks}")
logger.debug(f"Embedding vector shape: {embedding.shape}")
logger.debug(f"Similarity scores: {scores}")
```

**In our project**: Not currently used, but useful for:
- Tracking each step of document processing
- Logging intermediate values
- Debugging retrieval issues

#### INFO
**Use for**: General informational messages about normal operation
**Example**:
```python
logger.info("Starting document ingestion...")
logger.info(f"Successfully ingested document with {len(chunks)} chunks")
logger.info(f"Processing query: {request.question}")
```

**In our project**: Used extensively for:
- Document processing milestones
- Successful operations
- Query processing

#### WARNING
**Use for**: Something unexpected happened, but the system can continue
**Example**:
```python
logger.warning("No text content found in document")
logger.warning("Empty text provided for chunking")
logger.warning("Vector store not initialized, using default")
```

**In our project**: Used for:
- Non-critical issues
- Missing optional data
- Fallback scenarios

#### ERROR
**Use for**: Serious problems that prevent a function from completing
**Example**:
```python
logger.error(f"Error processing document: {str(e)}")
logger.error(f"Error creating vector store: {str(e)}")
logger.error(f"Error querying RAG pipeline: {str(e)}")
```

**In our project**: Used for:
- Exception handling
- Failed operations
- API errors

#### CRITICAL
**Use for**: Very serious errors that might cause the program to stop
**Example**:
```python
logger.critical("Database connection lost")
logger.critical("API key invalid - system cannot function")
logger.critical("Out of memory - shutting down")
```

**In our project**: Not currently used, but should be used for:
- System-level failures
- Security breaches
- Data corruption

### 4.3 Visual Guide

```
DEBUG   → 🔍 "I'm looking at chunk 5 of 10"
INFO    → ℹ️  "Document ingested successfully"
WARNING → ⚠️  "No text found, but continuing..."
ERROR   → ❌ "Failed to process document"
CRITICAL→ 🚨 "System cannot continue!"
```

---

## 5. Logging Components Explained

### 5.1 Logger

**What it is**: The main object you use to log messages

**How to create**:
```python
# Method 1: Use module name (RECOMMENDED)
logger = logging.getLogger(__name__)
# This creates a logger named after your module
# Example: "src.document_processor"

# Method 2: Use custom name
logger = logging.getLogger("my_custom_logger")

# Method 3: Use class name
logger = logging.getLogger(__class__.__name__)
```

**Why use `__name__`?**
- Automatically uses the module name
- Creates a hierarchy (parent.child loggers)
- Easy to filter logs by module

**Example from our code**:
```python
# In src/document_processor.py
logger = logging.getLogger(__name__)
# Logger name: "src.document_processor"

# In src/api.py
logger = logging.getLogger(__name__)
# Logger name: "src.api"
```

### 5.2 Handler

**What it is**: Determines where logs go

**Common handlers**:

**1. StreamHandler** (console output):
```python
handler = logging.StreamHandler()  # Goes to console
```

**2. FileHandler** (file output):
```python
handler = logging.FileHandler('app.log')  # Goes to file
```

**3. RotatingFileHandler** (file with rotation):
```python
from logging.handlers import RotatingFileHandler
handler = RotatingFileHandler(
    'app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5  # Keep 5 backup files
)
```

**4. Multiple handlers**:
```python
# Log to both console and file
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('app.log')

logger.addHandler(console_handler)
logger.addHandler(file_handler)
```

**In our project**: We use `basicConfig()` which creates a StreamHandler by default (console output).

### 5.3 Formatter

**What it is**: Controls how log messages are displayed

**Basic format string**:
```python
format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

**Format codes**:
- `%(asctime)s` - Timestamp
- `%(name)s` - Logger name
- `%(levelname)s` - Log level (INFO, ERROR, etc.)
- `%(message)s` - The actual message
- `%(filename)s` - Source filename
- `%(lineno)d` - Line number
- `%(funcName)s` - Function name
- `%(pathname)s` - Full pathname

**Example formats**:

**Simple**:
```python
format="%(levelname)s: %(message)s"
# Output: INFO: Document processed
```

**Detailed** (our current format):
```python
format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# Output: 2024-01-10 14:30:15,123 - src.api - INFO - Processing query
```

**Very detailed**:
```python
format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
# Output: 2024-01-10 14:30:15,123 [INFO] src.api:194 - Processing query
```

### 5.4 Filter

**What it is**: Controls which log records are processed

**Example**: Only log errors from a specific module
```python
class ErrorFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.ERROR

handler.addFilter(ErrorFilter())
```

---

## 6. How Logging Works in Our Code

### 6.1 Step-by-Step: What Happens When We Log

**Example**: `logger.info("Processing document...")`

1. **Logger receives message**:
   ```python
   logger.info("Processing document...")
   ```

2. **Logger checks level**:
   - Is INFO >= current level (INFO)? Yes → Continue
   - If level was WARNING, this would be filtered out

3. **Logger creates LogRecord**:
   - Includes: message, level, timestamp, module name, etc.

4. **Logger passes to handlers**:
   - Our handler is StreamHandler (console)

5. **Handler formats message**:
   - Uses our format: `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`

6. **Handler outputs**:
   - Writes to console: `2024-01-10 14:30:15,123 - src.api - INFO - Processing document...`

### 6.2 Real Example from Our Code

**File**: `src/api.py`, line 108

```python
logger.info("Starting document ingestion...")
```

**What happens**:
1. Logger name: `src.api`
2. Level: INFO
3. Message: "Starting document ingestion..."
4. Format applied
5. Output: `2024-01-10 14:30:15,123 - src.api - INFO - Starting document ingestion...`

### 6.3 Logger Hierarchy

Loggers can have parents and children:

```
root (top level)
  └── src
      ├── src.api
      ├── src.document_processor
      ├── src.chunker
      ├── src.vector_store
      └── src.rag_pipeline
```

**Benefits**:
- Can set level for entire `src` module
- Can filter logs by module
- Child loggers inherit parent settings

---

## 7. Advanced Logging Techniques

### 7.1 Structured Logging

**Instead of**:
```python
logger.info(f"User {user_id} queried {query}")
```

**Use**:
```python
logger.info("User queried system", extra={
    "user_id": user_id,
    "query": query,
    "timestamp": datetime.now().isoformat()
})
```

**Benefits**: Easier to parse, search, and analyze logs

### 7.2 Logging Exceptions

**Method 1: Using exc_info**:
```python
try:
    process_document()
except Exception as e:
    logger.error("Failed to process document", exc_info=True)
    # This automatically includes the full traceback
```

**Method 2: Using exception()**:
```python
try:
    process_document()
except Exception as e:
    logger.exception("Failed to process document")
    # Same as logger.error(..., exc_info=True)
```

**In our code** (current approach):
```python
except Exception as e:
    logger.error(f"Error processing document: {str(e)}")
    # Missing traceback! Should use exc_info=True
```

**Better approach**:
```python
except Exception as e:
    logger.exception("Error processing document")
    # Includes full traceback automatically
```

### 7.3 Contextual Logging

**Add context to all logs in a function**:
```python
import logging

def process_document(file_path):
    logger = logging.getLogger(__name__)
    
    # Add context
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.file_path = file_path
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    logger.info("Processing document")
    # All logs in this function will include file_path
```

### 7.4 Performance Logging

**Time operations**:
```python
import time
import logging

logger = logging.getLogger(__name__)

def process_document():
    start_time = time.time()
    
    # ... processing ...
    
    elapsed = time.time() - start_time
    logger.info(f"Document processed in {elapsed:.2f} seconds")
```

**Or use a decorator**:
```python
import functools
import time
import logging

logger = logging.getLogger(__name__)

def log_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

@log_time
def process_document():
    # ... processing ...
```

### 7.5 Conditional Logging

**Only log if condition is met**:
```python
# Instead of:
if debug_mode:
    logger.debug("Debug info")

# Use:
logger.debug("Debug info", extra={"debug_mode": True})
# Then filter in handler
```

**Or check level first**:
```python
if logger.isEnabledFor(logging.DEBUG):
    # Expensive operation only if DEBUG is enabled
    debug_data = expensive_debug_calculation()
    logger.debug(f"Debug data: {debug_data}")
```

---

## 8. Best Practices for RAG Systems

### 8.1 What to Log in RAG Systems

**Document Ingestion**:
- ✅ Start/end of ingestion
- ✅ Number of chunks created
- ✅ Document size
- ✅ Processing time
- ✅ Errors and warnings

**Query Processing**:
- ✅ Incoming queries
- ✅ Retrieved chunks (count and IDs)
- ✅ LLM response time
- ✅ Token usage (if available)
- ✅ Answer quality metrics

**Vector Operations**:
- ✅ Embedding generation time
- ✅ Similarity search time
- ✅ Number of results retrieved
- ✅ Similarity scores

**API Usage**:
- ✅ Request/response times
- ✅ Error rates
- ✅ Request patterns

### 8.2 Log Levels for RAG Operations

```python
# DEBUG: Detailed operation info
logger.debug(f"Processing chunk {i}/{total}")
logger.debug(f"Similarity score: {score:.4f}")

# INFO: Normal operations
logger.info("Document ingested successfully")
logger.info(f"Retrieved {len(chunks)} relevant chunks")

# WARNING: Non-critical issues
logger.warning("Low similarity scores for query")
logger.warning("Document has no extractable text")

# ERROR: Failures
logger.error("Failed to generate embeddings")
logger.error("LLM API call failed")

# CRITICAL: System failures
logger.critical("Vector database corrupted")
logger.critical("API key expired")
```

### 8.3 Sensitive Information

**❌ Don't log**:
```python
logger.info(f"API key: {api_key}")  # BAD!
logger.info(f"User password: {password}")  # BAD!
logger.info(f"Full document: {document_text}")  # Might be too verbose
```

**✅ Do log**:
```python
logger.info("API key validated")  # Good
logger.info(f"Document length: {len(document_text)}")  # Good
logger.info(f"Query received: {query[:50]}...")  # Truncated
```

---

## 9. Improving Logging in Our Project

### 9.1 Current Issues

**Issue 1**: Basic configuration in API file only
- **Problem**: Other modules might not have proper logging setup
- **Solution**: Create a centralized logging configuration

**Issue 2**: No file logging
- **Problem**: Logs are lost when server restarts
- **Solution**: Add file handler

**Issue 3**: No exception tracebacks
- **Problem**: Hard to debug errors
- **Solution**: Use `logger.exception()` instead of `logger.error()`

**Issue 4**: No performance logging
- **Problem**: Can't track slow operations
- **Solution**: Add timing logs

### 9.2 Improved Logging Configuration

**Create**: `src/logging_config.py`

```python
"""Centralized logging configuration."""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging configuration for the entire application.
    
    Args:
        log_level: Minimum log level to display
        log_file: Path to log file (optional)
    """
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler (always add)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    
    # File handler (if log_file specified)
    handlers = [console_handler]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # More detailed in file
        file_handler.setFormatter(detailed_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set to lowest, handlers filter
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set specific logger levels
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
```

**Update**: `src/api.py`

```python
from src.logging_config import setup_logging

# At the top, before other imports
setup_logging(
    log_level=logging.INFO,
    log_file='logs/rag_system.log'
)

logger = logging.getLogger(__name__)
```

### 9.3 Adding Performance Logging

**Add timing to document processing**:

```python
import time

def extract_text(self) -> str:
    start_time = time.time()
    try:
        # ... existing code ...
        elapsed = time.time() - start_time
        logger.info(
            f"Extracted {len(full_text)} characters in {elapsed:.2f}s"
        )
        return full_text
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"Failed to extract text after {elapsed:.2f}s: {str(e)}",
            exc_info=True
        )
        raise
```

### 9.4 Better Error Logging

**Current**:
```python
except Exception as e:
    logger.error(f"Error processing document: {str(e)}")
```

**Improved**:
```python
except FileNotFoundError as e:
    logger.error(f"Document not found: {self.file_path}", exc_info=True)
    raise
except Exception as e:
    logger.exception("Unexpected error processing document")
    raise
```

### 9.5 Query Logging Enhancement

**Add more context to query logs**:

```python
@app.post("/query")
async def query_document(request: QueryRequest):
    query_start = time.time()
    
    logger.info(
        f"Query received: '{request.question[:100]}...' "
        f"(length: {len(request.question)})"
    )
    
    try:
        result = rag_pipeline.query(request.question)
        
        elapsed = time.time() - query_start
        logger.info(
            f"Query answered in {elapsed:.2f}s. "
            f"Retrieved {len(result['sources'])} sources."
        )
        
        return result
    except Exception as e:
        elapsed = time.time() - query_start
        logger.error(
            f"Query failed after {elapsed:.2f}s: {str(e)}",
            exc_info=True
        )
        raise
```

---

## 10. Practical Examples

### 10.1 Example 1: Document Processing with Full Logging

```python
import logging
import time

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def extract_text(self, file_path: str) -> str:
        logger.info(f"Starting extraction from: {file_path}")
        start_time = time.time()
        
        try:
            doc = Document(file_path)
            logger.debug(f"Document opened successfully")
            
            text_parts = []
            paragraph_count = 0
            table_count = 0
            
            # Extract paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())
                    paragraph_count += 1
            
            logger.debug(f"Extracted {paragraph_count} paragraphs")
            
            # Extract tables
            for table in doc.tables:
                table_count += 1
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_text:
                        text_parts.append(" | ".join(row_text))
            
            logger.debug(f"Extracted {table_count} tables")
            
            full_text = "\n\n".join(text_parts)
            elapsed = time.time() - start_time
            
            if not full_text.strip():
                logger.warning(f"No text content found in {file_path}")
                return ""
            
            logger.info(
                f"Extraction complete: {len(full_text)} chars, "
                f"{paragraph_count} paragraphs, {table_count} tables "
                f"in {elapsed:.2f}s"
            )
            
            return full_text
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}", exc_info=True)
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(
                f"Error extracting text from {file_path} "
                f"(failed after {elapsed:.2f}s)"
            )
            raise
```

### 10.2 Example 2: Vector Store with Detailed Logging

```python
def create_collection(self, texts: List[str], metadatas: Optional[List[dict]] = None):
    logger.info(f"Creating vector collection with {len(texts)} documents")
    start_time = time.time()
    
    if not texts:
        logger.warning("No texts provided for vector store")
        return
    
    try:
        logger.debug(f"Using embedding model: {self.embedding_model}")
        logger.debug(f"Persist directory: {self.persist_directory}")
        logger.debug(f"Collection name: {self.collection_name}")
        
        # Create embeddings (this might take time)
        embedding_start = time.time()
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            metadatas=metadatas or [{}] * len(texts)
        )
        embedding_time = time.time() - embedding_start
        
        total_time = time.time() - start_time
        
        logger.info(
            f"Vector store created: {len(texts)} documents, "
            f"embeddings generated in {embedding_time:.2f}s, "
            f"total time {total_time:.2f}s"
        )
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(
            f"Error creating vector store after {elapsed:.2f}s"
        )
        raise
```

### 10.3 Example 3: RAG Pipeline with Query Logging

```python
def query(self, question: str) -> dict:
    logger.info(f"Processing query: '{question[:100]}...'")
    query_start = time.time()
    
    if not self.qa_chain:
        logger.error("RAG pipeline not initialized")
        raise ValueError("RAG pipeline not initialized")
    
    try:
        # Retrieve relevant chunks
        retrieval_start = time.time()
        retriever = self.vector_store.get_retriever(k=self.top_k)
        retrieval_time = time.time() - retrieval_start
        
        logger.debug(f"Retrieval setup took {retrieval_time:.3f}s")
        
        # Generate answer
        generation_start = time.time()
        result = self.qa_chain({"query": question})
        generation_time = time.time() - generation_start
        
        total_time = time.time() - query_start
        
        sources_count = len(result.get("source_documents", []))
        
        logger.info(
            f"Query answered in {total_time:.2f}s "
            f"(retrieval: {retrieval_time:.3f}s, "
            f"generation: {generation_time:.2f}s, "
            f"sources: {sources_count})"
        )
        
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
        elapsed = time.time() - query_start
        logger.exception(
            f"Error querying RAG pipeline after {elapsed:.2f}s"
        )
        raise
```

### 10.4 Example 4: Complete Logging Setup

**File**: `src/logging_config.py`

```python
"""Complete logging setup for RAG system."""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

def setup_logging(
    log_level=logging.INFO,
    log_dir="logs",
    log_file="rag_system.log",
    enable_file_logging=True
):
    """
    Set up comprehensive logging for the RAG system.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        log_file: Name of log file
        enable_file_logging: Whether to log to file
    """
    # Create log directory
    if enable_file_logging:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        full_log_path = log_path / log_file
    
    # Detailed formatter for files
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Simple formatter for console
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    handlers = [console_handler]
    
    # File handler with rotation
    if enable_file_logging:
        file_handler = RotatingFileHandler(
            full_log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # More verbose in file
        file_handler.setFormatter(detailed_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set to lowest, handlers will filter
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    # Our application loggers
    app_logger = logging.getLogger('src')
    app_logger.setLevel(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, file={full_log_path if enable_file_logging else 'disabled'}")
```

**Usage in main.py**:
```python
from src.logging_config import setup_logging
import logging

# Set up logging first
setup_logging(
    log_level=logging.INFO,
    enable_file_logging=True
)

logger = logging.getLogger(__name__)
logger.info("Starting RAG system...")
```

---

## Summary

### Key Takeaways

1. **Always use logging, never print()** for production code
2. **Use appropriate log levels**: DEBUG for details, INFO for normal operations, ERROR for failures
3. **Include context**: Timestamps, module names, function names
4. **Log exceptions properly**: Use `logger.exception()` to include tracebacks
5. **Log performance**: Track timing for slow operations
6. **Don't log sensitive data**: API keys, passwords, full documents
7. **Use structured logging**: For easier parsing and analysis
8. **Centralize configuration**: One place to configure all logging

### Quick Reference

```python
# Basic setup
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Logging methods
logger.debug("Detailed info")
logger.info("Normal operation")
logger.warning("Something unexpected")
logger.error("Error occurred")
logger.critical("Critical failure")

# Logging exceptions
try:
    risky_operation()
except Exception:
    logger.exception("Operation failed")  # Includes traceback

# Performance logging
import time
start = time.time()
operation()
logger.info(f"Operation took {time.time() - start:.2f}s")
```

### Next Steps

1. Review current logging in each file
2. Add `exc_info=True` to error logs
3. Add performance timing logs
4. Create centralized logging configuration
5. Add file logging with rotation
6. Add structured logging for queries

Happy logging! 📝
