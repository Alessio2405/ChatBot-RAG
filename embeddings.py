import ollama
import numpy as np
from typing import List, Optional
import time

class EmbeddingManager:
    def __init__(self, model_name: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the Ollama model to use for embeddings
            base_url: URL of the Ollama server
        """
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)
        
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Call Ollama embedding API
            response = self.client.embeddings(model=self.model_name, prompt=text)
            
            # Convert to numpy array
            embedding = np.array(response['embedding'], dtype=np.float32)
            
            return embedding
        except Exception as e:
            raise Exception(f"Error getting embedding: {str(e)}")
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for text in batch:
                try:
                    embedding = self.get_embedding(text)
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Error embedding text: {str(e)}")
                    # Add a zero vector as fallback
                    embeddings.append(np.zeros(4096, dtype=np.float32))
            
            # Small delay between batches to avoid overwhelming the server
            if i + batch_size < len(texts):
                time.sleep(0.1)
        
        return embeddings
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def l2_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate L2 distance between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            L2 distance
        """
        return np.linalg.norm(vec1 - vec2)
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize an embedding vector to unit length.
        
        Args:
            embedding: Input embedding vector
            
        Returns:
            Normalized embedding vector
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def test_connection(self) -> bool:
        """
        Test if the Ollama server is accessible.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to get a simple embedding
            test_embedding = self.get_embedding("test")
            return len(test_embedding) > 0
        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            return False
    
    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        try:
            # This would require additional Ollama API calls
            # For now, return basic info
            return {
                'model_name': self.model_name,
                'base_url': self.client.host,
                'connection_status': self.test_connection()
            }
        except Exception as e:
            return {
                'model_name': self.model_name,
                'base_url': self.client.host,
                'connection_status': False,
                'error': str(e)
            } 