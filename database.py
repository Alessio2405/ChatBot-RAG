import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional
import threading

class DatabaseManager:
    def __init__(self, db_path: str = "documents.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize the database with required tables."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create chunks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
                )
            ''')
            
            # Create chats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT NOT NULL,
                    bot_output TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
    
    def add_document(self, file_name: str) -> int:
        """Add a new document and return its ID."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents (file_name) VALUES (?)",
                (file_name,)
            )
            doc_id = cursor.lastrowid
            conn.commit()
            conn.close()
            if doc_id is None:
                raise RuntimeError("Failed to insert document")
            return doc_id
    
    def add_chunk(self, doc_id: int, text: str, embedding: np.ndarray):
        """Add a text chunk with its embedding."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Convert numpy array to bytes for storage
            embedding_bytes = embedding.tobytes()
            cursor.execute(
                "INSERT INTO chunks (doc_id, text, embedding) VALUES (?, ?, ?)",
                (doc_id, text, embedding_bytes)
            )
            conn.commit()
            conn.close()
    
    def add_chat(self, user_input: str, bot_output: str):
        """Add a chat interaction."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chats (user_input, bot_output) VALUES (?, ?)",
                (user_input, bot_output)
            )
            conn.commit()
            conn.close()
    
    def get_all_chunks(self) -> List[Tuple[int, str, np.ndarray]]:
        """Retrieve all chunks with their embeddings."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_id, text, embedding FROM chunks")
            results = cursor.fetchall()
            conn.close()
            
            chunks = []
            for chunk_id, text, embedding_bytes in results:
                # Convert bytes back to numpy array
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                chunks.append((chunk_id, text, embedding))
            
            return chunks
    
    def get_document_chunks(self, doc_id: int) -> List[Tuple[int, str, np.ndarray]]:
        """Retrieve chunks for a specific document."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT chunk_id, text, embedding FROM chunks WHERE doc_id = ?",
                (doc_id,)
            )
            results = cursor.fetchall()
            conn.close()
            
            chunks = []
            for chunk_id, text, embedding_bytes in results:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                chunks.append((chunk_id, text, embedding))
            
            return chunks
    
    def get_chat_history(self, limit: int = 10) -> List[Tuple[str, str, str]]:
        """Retrieve recent chat history."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_input, bot_output, timestamp FROM chats ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            results = cursor.fetchall()
            conn.close()
            return results
    
    def get_documents(self) -> List[Tuple[int, str, str]]:
        """Retrieve all uploaded documents."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT doc_id, file_name, uploaded_at FROM documents ORDER BY uploaded_at DESC")
            results = cursor.fetchall()
            conn.close()
            return results
    
    def delete_document(self, doc_id: int):
        """Delete a document and all its chunks."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Delete chunks first due to foreign key constraint
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()
            conn.close() 