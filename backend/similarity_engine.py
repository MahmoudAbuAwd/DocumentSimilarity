"""
Document Similarity Engine using all-MiniLM-L6-v2
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import config

class SimilarityEngine:
    def __init__(self, model_name: str = None):
        """Initialize the similarity engine."""
        self.model_name = model_name or config.MODEL_NAME
        self.model = None
        self.embeddings_cache_path = config.EMBEDDINGS_CACHE
        
    def _load_model(self):
        """Lazy load the model when needed."""
        if self.model is None:
            print(f"🤖 Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("✅ Model loaded successfully")
    
    def generate_embeddings(self, documents: List[Dict]) -> np.ndarray:
        """Generate embeddings for documents."""
        self._load_model()
        
        print(f"🔢 Generating embeddings for {len(documents)} documents...")
        
        # Extract text content
        texts = [doc['content'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Cache embeddings if enabled
        if config.SAVE_EMBEDDINGS:
            self._cache_embeddings(embeddings, documents)
        
        return embeddings
    
    def _cache_embeddings(self, embeddings: np.ndarray, documents: List[Dict]):
        """Cache embeddings to disk for faster future processing."""
        cache_data = {
            'embeddings': embeddings,
            'documents': [
                {
                    'name': doc['name'],
                    'path': doc['path'],
                    'size': doc['size']
                } for doc in documents
            ],
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'embedding_dimension': embeddings.shape[1]
        }
        
        self.embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.embeddings_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"💾 Embeddings cached to: {self.embeddings_cache_path}")
    
    def load_cached_embeddings(self, documents: List[Dict]) -> Optional[np.ndarray]:
        """Try to load cached embeddings if they match current documents."""
        if not self.embeddings_cache_path.exists():
            return None
        
        try:
            with open(self.embeddings_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if cache is valid
            if (cache_data.get('model_name') != self.model_name or
                len(cache_data.get('documents', [])) != len(documents)):
                return None
            
            # Check if documents match (simple name and size check)
            cached_docs = {doc['name']: doc['size'] for doc in cache_data['documents']}
            current_docs = {doc['name']: doc['size'] for doc in documents}
            
            if cached_docs != current_docs:
                return None
            
            print("✅ Using cached embeddings")
            return cache_data['embeddings']
            
        except Exception as e:
            print(f"⚠️ Could not load cached embeddings: {e}")
            return None
    
    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate pairwise cosine similarities."""
        print("📊 Computing similarity matrix...")
        return cosine_similarity(embeddings)
    
    def find_similarities(self, documents: List[Dict]) -> Dict:
        """Find similarities between all document pairs."""
        if len(documents) < 2:
            return {
                "error": "Need at least 2 documents for comparison",
                "total_documents": len(documents)
            }
        
        print(f"🔍 Finding similarities for {len(documents)} documents")
        
        # Try to load cached embeddings first
        embeddings = self.load_cached_embeddings(documents)
        
        if embeddings is None:
            # Generate new embeddings
            embeddings = self.generate_embeddings(documents)
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(embeddings)
        
        # Extract similarity pairs
        similarities = []
        n_docs = len(documents)
        
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                score = float(similarity_matrix[i][j])
                
                if score >= config.MIN_SIMILARITY_THRESHOLD:
                    similarities.append({
                        'doc1': documents[i]['name'],
                        'doc2': documents[j]['name'],
                        'score': score,
                        'doc1_path': documents[i]['path'],
                        'doc2_path': documents[j]['path'],
                        'doc1_size': documents[i]['size'],
                        'doc2_size': documents[j]['size']
                    })
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit results if configured
        if config.TOP_RESULTS_LIMIT:
            similarities = similarities[:config.TOP_RESULTS_LIMIT]
        
        # Prepare comprehensive results
        results = {
            'metadata': {
                'model_used': self.model_name,
                'embedding_dimension': embeddings.shape[1],
                'timestamp': datetime.now().isoformat(),
                'total_documents': len(documents),
                'total_pairs_analyzed': n_docs * (n_docs - 1) // 2,
                'pairs_above_threshold': len(similarities),
                'min_threshold': config.MIN_SIMILARITY_THRESHOLD
            },
            'documents': [
                {
                    'name': doc['name'],
                    'path': doc['path'],
                    'size': doc['size'],
                    'extension': doc['extension']
                } for doc in documents
            ],
            'similarities': similarities,
            'statistics': self._calculate_statistics(similarities) if similarities else {}
        }
        
        # Save results to file
        self._save_results(results)
        
        return results
    
    def find_most_similar_to_query(self, query_text: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Find documents most similar to a query text."""
        self._load_model()
        
        # Create a temporary document for the query
        query_doc = {
            'name': 'query',
            'content': query_text,
            'path': 'query',
            'size': len(query_text),
            'extension': '.txt'
        }
        
        all_docs = [query_doc] + documents
        embeddings = self.generate_embeddings(all_docs)
        
        query_embedding = embeddings[0:1]  # First embedding is query
        doc_embeddings = embeddings[1:]    # Rest are document embeddings
        
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Get top-k similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= config.MIN_SIMILARITY_THRESHOLD:
                results.append({
                    'document': documents[idx]['name'],
                    'score': float(similarities[idx]),
                    'path': documents[idx]['path'],
                    'size': documents[idx]['size'],
                    'extension': documents[idx]['extension']
                })
        
        return results
    
    def _calculate_statistics(self, similarities: List[Dict]) -> Dict:
        """Calculate basic statistics about similarity scores."""
        if not similarities:
            return {}
        
        scores = [sim['score'] for sim in similarities]
        
        return {
            'mean_similarity': float(np.mean(scores)),
            'max_similarity': float(np.max(scores)),
            'min_similarity': float(np.min(scores)),
            'std_similarity': float(np.std(scores)),
            'median_similarity': float(np.median(scores))
        }
    
    def _save_results(self, results: Dict):
        """Save results to JSON file."""
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(config.RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {config.RESULTS_FILE}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from the model."""
        self._load_model()
        return self.model.get_sentence_embedding_dimension()
    
    def clear_cache(self):
        """Clear cached embeddings."""
        if self.embeddings_cache_path.exists():
            self.embeddings_cache_path.unlink()
            print("🗑️ Embedding cache cleared")