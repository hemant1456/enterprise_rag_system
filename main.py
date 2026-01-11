"""Main entry point for the Enterprise RAG System."""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Configure logging BEFORE importing other modules
log_file = Path("enterprise_rag.log")
log_file.parent.mkdir(exist_ok=True)

# Create formatters
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

console_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)

# File handler with rotation
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(detailed_formatter)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler, file_handler],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Reduce noise from third-party libraries
logging.getLogger('uvicorn').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Now import after logging is configured
import uvicorn
from config import settings

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Starting Enterprise RAG System")
    logger.info(f"Server: http://{settings.api_host}:{settings.api_port}")
    logger.info(f"Log file: {log_file.absolute()}")
    logger.info("="*60)
    
    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
