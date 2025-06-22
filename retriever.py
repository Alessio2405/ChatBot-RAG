from typing import List, Tuple, Optional
from database import DatabaseManager
from embeddings import EmbeddingManager

class Retriever:
    def __init__(self, db_manager: DatabaseManager, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever.
        
        Args:
            db_manager: Database manager instance
            embedding_manager: Embedding manager instance
        """
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
    
    def retrieve_similar_chunks(self, query: str, top_k: int = 5, similarity_threshold: float = 0.5) -> List[Tuple[int, str, float]]:
        """
        Retrieve the most similar chunks for a given query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score to include results
            
        Returns:
            List of tuples (chunk_id, text, similarity_score)
        """
        try:
            # Get query embedding
            query_embedding = self.embedding_manager.get_embedding(query)
            
            # Get all chunks from database
            chunks = self.db_manager.get_all_chunks()
            
            if not chunks:
                return []
            
            # Calculate similarities
            similarities = []
            for chunk_id, text, chunk_embedding in chunks:
                similarity = self.embedding_manager.cosine_similarity(query_embedding, chunk_embedding)
                similarities.append((chunk_id, text, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[2], reverse=True)
            
            # Filter by threshold and return top_k
            filtered_results = [
                (chunk_id, text, score) 
                for chunk_id, text, score in similarities 
                if score >= similarity_threshold
            ]
            
            return filtered_results[:top_k]
            
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []
    
    def retrieve_by_l2_distance(self, query: str, top_k: int = 5, max_distance: float = 2.0) -> List[Tuple[int, str, float]]:
        """
        Retrieve chunks using L2 distance.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            max_distance: Maximum L2 distance to include results
            
        Returns:
            List of tuples (chunk_id, text, distance)
        """
        try:
            # Get query embedding
            query_embedding = self.embedding_manager.get_embedding(query)
            
            # Get all chunks from database
            chunks = self.db_manager.get_all_chunks()
            
            if not chunks:
                return []
            
            # Calculate L2 distances
            distances = []
            for chunk_id, text, chunk_embedding in chunks:
                distance = self.embedding_manager.l2_distance(query_embedding, chunk_embedding)
                distances.append((chunk_id, text, distance))
            
            # Sort by distance (ascending)
            distances.sort(key=lambda x: x[2])
            
            # Filter by max_distance and return top_k
            filtered_results = [
                (chunk_id, text, distance) 
                for chunk_id, text, distance in distances 
                if distance <= max_distance
            ]
            
            return filtered_results[:top_k]
            
        except Exception as e:
            print(f"Error in L2 distance search: {str(e)}")
            return []
    
    def build_context_from_chunks(self, chunks: List[Tuple[int, str, float]], max_context_length: int = 2000) -> str:
        """
        Build context string from retrieved chunks.
        
        Args:
            chunks: List of (chunk_id, text, score) tuples
            max_context_length: Maximum length of context string
            
        Returns:
            Context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        current_length = 0
        
        for chunk_id, text, score in chunks:
            # Add chunk with score information
            chunk_with_score = f"[Score: {score:.3f}] {text}"
            
            if current_length + len(chunk_with_score) > max_context_length:
                break
            
            context_parts.append(chunk_with_score)
            current_length += len(chunk_with_score)
        
        return "\n\n".join(context_parts)
    
    def hybrid_search(self, query: str, top_k: int = 5, 
                     cosine_weight: float = 0.7, l2_weight: float = 0.3) -> List[Tuple[int, str, float]]:
        """
        Perform hybrid search using both cosine similarity and L2 distance.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            cosine_weight: Weight for cosine similarity
            l2_weight: Weight for L2 distance
            
        Returns:
            List of tuples (chunk_id, text, combined_score)
        """
        try:
            # Get both similarity and distance results
            cosine_results = self.retrieve_similar_chunks(query, top_k * 2, 0.0)
            l2_results = self.retrieve_by_l2_distance(query, top_k * 2, float('inf'))
            
            # Create dictionaries for easy lookup
            cosine_dict = {chunk_id: score for chunk_id, _, score in cosine_results}
            l2_dict = {chunk_id: distance for chunk_id, _, distance in l2_results}
            
            # Get all unique chunk IDs
            all_chunk_ids = set(cosine_dict.keys()) | set(l2_dict.keys())
            
            # Calculate combined scores
            combined_scores = []
            for chunk_id in all_chunk_ids:
                cosine_score = cosine_dict.get(chunk_id, 0.0)
                l2_distance = l2_dict.get(chunk_id, float('inf'))
                
                # Normalize L2 distance to 0-1 range (assuming max distance of 2.0)
                l2_score = max(0, 1 - (l2_distance / 2.0))
                
                # Calculate combined score
                combined_score = (cosine_weight * cosine_score) + (l2_weight * l2_score)
                
                # Get text from either result
                text = next((text for cid, text, _ in cosine_results if cid == chunk_id), 
                           next((text for cid, text, _ in l2_results if cid == chunk_id), ""))
                
                combined_scores.append((chunk_id, text, combined_score))
            
            # Sort by combined score and return top_k
            combined_scores.sort(key=lambda x: x[2], reverse=True)
            return combined_scores[:top_k]
            
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            return []
    
    def get_document_statistics(self) -> dict:
        """
        Get statistics about stored documents and chunks.
        
        Returns:
            Dictionary with statistics
        """
        try:
            documents = self.db_manager.get_documents()
            chunks = self.db_manager.get_all_chunks()
            
            return {
                'total_documents': len(documents),
                'total_chunks': len(chunks),
                'documents': [
                    {
                        'doc_id': doc_id,
                        'file_name': file_name,
                        'uploaded_at': uploaded_at
                    }
                    for doc_id, file_name, uploaded_at in documents
                ]
            }
        except Exception as e:
            return {
                'error': str(e),
                'total_documents': 0,
                'total_chunks': 0,
                'documents': []
            } 