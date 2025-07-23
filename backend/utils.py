"""
Utility functions for document similarity project
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def ensure_directory(directory_path: Path) -> bool:
    """Ensure directory exists, create if it doesn't."""
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

def save_json(data: Dict[str, Any], file_path: Path) -> bool:
    """Save data to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False

def load_json(file_path: Path) -> Dict[str, Any]:
    """Load data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return {}

def calculate_similarity_statistics(similarities: List[Dict]) -> Dict[str, float]:
    """Calculate statistics for similarity scores."""
    if not similarities:
        return {}
    
    scores = [sim['score'] for sim in similarities]
    scores_array = np.array(scores)
    
    return {
        'count': len(scores),
        'mean': float(np.mean(scores_array)),
        'median': float(np.median(scores_array)),
        'std': float(np.std(scores_array)),
        'min': float(np.min(scores_array)),
        'max': float(np.max(scores_array)),
        'q25': float(np.percentile(scores_array, 25)),
        'q75': float(np.percentile(scores_array, 75))
    }

def format_timestamp(timestamp: str = None) -> str:
    """Format timestamp for display."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def validate_similarity_score(score: float) -> bool:
    """Validate that similarity score is in valid range."""
    return 0.0 <= score <= 1.0

def get_file_extension(filename: str) -> str:
    """Get file extension from filename."""
    return Path(filename).suffix.lower()

def clean_filename(filename: str) -> str:
    """Clean filename for safe storage."""
    import re
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove excessive dots
    filename = re.sub(r'\.+', '.', filename)
    return filename.strip()

def memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

def progress_callback(current: int, total: int, description: str = "") -> None:
    """Simple progress callback for console output."""
    percentage = (current / total) * 100
    print(f"\r{description} Progress: {current}/{total} ({percentage:.1f}%)", end="", flush=True)
    if current == total:
        print()  # New line when complete