"""
Configuration settings for Document Similarity Engine
"""

import os
from pathlib import Path

# Model Configuration
MODEL_NAME = "all-MiniLM-L6-v2"

# Directory Paths (relative to backend folder)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input" / "documents"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR = BASE_DIR.parent / "frontend" / "static" / "uploads"

# File Paths
EMBEDDINGS_CACHE = PROCESSED_DIR / "embeddings.pkl"
RESULTS_FILE = RESULTS_DIR / "similarity_results.json"

# Processing Settings
MIN_SIMILARITY_THRESHOLD = 0.1  # Minimum similarity score to include
MAX_DOCUMENT_LENGTH = 50000     # Maximum characters per document
CHUNK_SIZE = 512               # Characters per chunk for large documents

# Supported File Extensions
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.md', '.rtf']

# API Settings
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md', 'rtf'}

# Output Settings
SAVE_EMBEDDINGS = True         # Cache embeddings for faster reprocessing
INCLUDE_METADATA = True        # Include file metadata in results
TOP_RESULTS_LIMIT = 20         # Maximum number of similarity pairs to return

# Create directories if they don't exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [DATA_DIR, INPUT_DIR, PROCESSED_DIR, RESULTS_DIR, UPLOAD_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories when config is imported
ensure_directories()