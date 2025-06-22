import PyPDF2
import io
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

class FileProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the file processor.
        
        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_file: File-like object containing PDF data
            
        Returns:
            Extracted text as string
        """
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_from_txt(self, txt_file) -> str:
        """
        Extract text from a TXT file.
        
        Args:
            txt_file: File-like object containing text data
            
        Returns:
            Extracted text as string
        """
        try:
            # Read the file content
            content = txt_file.read()
            
            # Try to decode as UTF-8, fallback to other encodings if needed
            if isinstance(content, bytes):
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        text = content.decode('latin-1')
                    except UnicodeDecodeError:
                        text = content.decode('cp1252', errors='ignore')
            else:
                text = str(content)
            
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from TXT file: {str(e)}")
    
    def extract_text(self, file) -> str:
        """
        Extract text from a file based on its type.
        
        Args:
            file: StreamlitUploadedFile object
            
        Returns:
            Extracted text as string
        """
        file_extension = file.name.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file)
        elif file_extension == 'txt':
            return self.extract_text_from_txt(file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Use LangChain's text splitter for better chunking
        chunks = self.text_splitter.split_text(text)
        
        # Filter out empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        return chunks
    
    def process_file(self, file) -> Tuple[str, List[str]]:
        """
        Process a file: extract text and chunk it.
        
        Args:
            file: StreamlitUploadedFile object
            
        Returns:
            Tuple of (extracted_text, list_of_chunks)
        """
        # Extract text from file
        text = self.extract_text(file)
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        return text, chunks
    
    def get_file_info(self, file) -> dict:
        """
        Get basic information about a file.
        
        Args:
            file: StreamlitUploadedFile object
            
        Returns:
            Dictionary with file information
        """
        return {
            'name': file.name,
            'size': file.size,
            'type': file.type
        } 