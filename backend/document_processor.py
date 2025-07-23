"""
Document processing utilities for handling different file formats
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
import docx
import config

class DocumentProcessor:
    def __init__(self):
        self.supported_extensions = config.SUPPORTED_EXTENSIONS
        
    def load_documents(self, directory_path: Path) -> List[Dict]:
        """Load all supported documents from a directory."""
        documents = []
        
        if not directory_path.exists():
            print(f"Directory not found: {directory_path}")
            return documents
            
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    content = self.extract_text(file_path)
                    if content and content.strip():
                        documents.append({
                            'name': file_path.name,
                            'path': str(file_path),
                            'content': self.preprocess_text(content),
                            'size': file_path.stat().st_size,
                            'extension': file_path.suffix.lower()
                        })
                        print(f"✅ Loaded: {file_path.name}")
                    else:
                        print(f"⚠️ Empty content: {file_path.name}")
                except Exception as e:
                    print(f"❌ Error loading {file_path.name}: {str(e)}")
                    
        return documents
    
    def extract_text(self, file_path: Path) -> Optional[str]:
        """Extract text content from different file formats."""
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.txt' or extension == '.md':
                return self._extract_from_txt(file_path)
            elif extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif extension == '.docx':
                return self._extract_from_docx(file_path)
            else:
                print(f"Unsupported format: {extension}")
                return None
        except Exception as e:
            print(f"Error extracting from {file_path.name}: {str(e)}")
            return None
    
    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from .txt and .md files."""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode {file_path.name} with any supported encoding")
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from Word documents."""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text content."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Limit document length
        if len(text) > config.MAX_DOCUMENT_LENGTH:
            text = text[:config.MAX_DOCUMENT_LENGTH]
            
        return text.strip()
    
    def chunk_document(self, text: str, chunk_size: int = None) -> List[str]:
        """Split large documents into smaller chunks."""
        if chunk_size is None:
            chunk_size = config.CHUNK_SIZE
            
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate if file can be processed."""
        if not file_path.exists():
            return False
            
        if file_path.suffix.lower() not in self.supported_extensions:
            return False
            
        if file_path.stat().st_size > config.MAX_FILE_SIZE:
            return False
            
        return True
    
    def get_file_info(self, file_path: Path) -> Dict:
        """Get metadata about a file."""
        if not file_path.exists():
            return {}
            
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'size': stat.st_size,
            'extension': file_path.suffix.lower(),
            'modified': stat.st_mtime,
            'is_supported': file_path.suffix.lower() in self.supported_extensions
        }