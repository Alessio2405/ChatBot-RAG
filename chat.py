import ollama
import json
from typing import Optional, List, Dict, Any, Generator
import time

class ChatManager:
    def __init__(self, model_name: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"):
        """
        Initialize the chat manager.
        
        Args:
            model_name: Name of the Ollama model to use for chat
            base_url: URL of the Ollama server
        """
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)
        self.conversation_history = []
    
    def chat(self, message: str, context: str = "", system_prompt: str = "") -> str:
        """
        Send a message to the chat model.
        
        Args:
            message: User message
            context: Context from retrieved documents (for RAG)
            system_prompt: System prompt to guide the model
            
        Returns:
            Model response
        """
        try:
            # Build the full prompt
            if context and system_prompt:
                full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser: {message}\nAssistant:"
            elif context:
                full_prompt = f"Context:\n{context}\n\nUser: {message}\nAssistant:"
            elif system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {message}\nAssistant:"
            else:
                full_prompt = f"User: {message}\nAssistant:"
            
            # Call Ollama chat API
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            # Extract the response content
            if response and 'message' in response:
                return response['message']['content']
            else:
                return "Sorry, I couldn't generate a response."
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chat_with_stream(self, message: str, context: str = "", system_prompt: str = "") -> Generator[str, None, None]:
        """
        Send a message to the chat model with streaming response.
        
        Args:
            message: User message
            context: Context from retrieved documents (for RAG)
            system_prompt: System prompt to guide the model
            
        Yields:
            Streamed model response chunks
        """
        try:
            # Build the full prompt
            if context and system_prompt:
                full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser: {message}\nAssistant:"
            elif context:
                full_prompt = f"Context:\n{context}\n\nUser: {message}\nAssistant:"
            elif system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {message}\nAssistant:"
            else:
                full_prompt = f"User: {message}\nAssistant:"
            
            # Call Ollama chat API with streaming
            stream = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}],
                stream=True
            )
            
            for chunk in stream:
                if chunk and 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    yield content
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield error_msg
    
    def chat_with_rag(self, message: str, retrieved_chunks: List[tuple], 
                     system_prompt: str = "You are a helpful assistant. Use the provided context to answer questions accurately.") -> str:
        """
        Chat with RAG (Retrieval-Augmented Generation).
        
        Args:
            message: User message
            retrieved_chunks: List of (chunk_id, text, score) tuples
            system_prompt: System prompt to guide the model
            
        Returns:
            Model response
        """
        if not retrieved_chunks:
            return self.chat(message, system_prompt=system_prompt)
        
        # Build context from retrieved chunks
        context_parts = []
        for chunk_id, text, score in retrieved_chunks:
            context_parts.append(f"[Relevance: {score:.3f}] {text}")
        
        context = "\n\n".join(context_parts)
        
        return self.chat(message, context, system_prompt)
    
    def chat_with_rag_stream(self, message: str, retrieved_chunks: List[tuple], 
                           system_prompt: str = "You are a helpful assistant. Use the provided context to answer questions accurately.") -> Generator[str, None, None]:
        """
        Chat with RAG using streaming response.
        
        Args:
            message: User message
            retrieved_chunks: List of (chunk_id, text, score) tuples
            system_prompt: System prompt to guide the model
            
        Yields:
            Streamed model response
        """
        if not retrieved_chunks:
            yield from self.chat_with_stream(message, system_prompt=system_prompt)
            return
        
        # Build context from retrieved chunks
        context_parts = []
        for chunk_id, text, score in retrieved_chunks:
            context_parts.append(f"[Relevance: {score:.3f}] {text}")
        
        context = "\n\n".join(context_parts)
        
        yield from self.chat_with_stream(message, context, system_prompt)
    
    def add_to_history(self, user_input: str, bot_output: str):
        """
        Add a conversation turn to history.
        
        Args:
            user_input: User message
            bot_output: Bot response
        """
        self.conversation_history.append({
            'user': user_input,
            'assistant': bot_output,
            'timestamp': time.time()
        })
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history.
        
        Args:
            limit: Maximum number of turns to return
            
        Returns:
            List of conversation turns
        """
        return self.conversation_history[-limit:] if self.conversation_history else []
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def test_connection(self) -> bool:
        """
        Test if the Ollama server is accessible.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to send a simple message
            response = self.chat("Hello")
            return response and "Error:" not in response and len(response) > 0
        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            return False
    
    def get_model_info(self) -> dict:
        """
        Get information about the chat model.
        
        Returns:
            Dictionary with model information
        """
        try:
            return {
                'model_name': self.model_name,
                'base_url': self.client.host,
                'connection_status': self.test_connection(),
                'history_length': len(self.conversation_history)
            }
        except Exception as e:
            return {
                'model_name': self.model_name,
                'base_url': self.client.host,
                'connection_status': False,
                'error': str(e),
                'history_length': len(self.conversation_history)
            }
    
    def get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for the assistant.
        
        Returns:
            Default system prompt
        """
        return """You are a helpful AI assistant. You can help users with various tasks including:

1. Answering questions based on provided context
2. Analyzing documents and extracting information
3. Providing explanations and insights
4. Helping with research and analysis

When using retrieved context:
- Base your answers on the provided context
- If the context doesn't contain relevant information, say so
- Cite specific parts of the context when appropriate
- Be accurate and honest about what you know

Always be helpful, accurate, and respectful in your responses.""" 