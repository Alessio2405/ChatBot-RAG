# Document Chat with RAG

A complete Streamlit application that lets users upload PDF and TXT files, persist them in a local SQLite database, embed their contents using Ollama's Qwen models, and chat with the documents using retrieval-augmented generation (RAG).

## Features

- 📤 **File Upload**: Support for PDF and TXT files with multi-file upload
- 🔍 **Text Processing**: Automatic text extraction and intelligent chunking
- 🧠 **Embeddings**: Vector embeddings using Ollama's Qwen models
- 💾 **Local Storage**: SQLite database for persistent document and chat storage
- 🔍 **RAG**: Retrieval-augmented generation with similarity search
- 💬 **Chat Interface**: Interactive chat with streaming responses
- 📊 **Database Management**: View and manage uploaded documents and chat history

## Demonstration Video
https://github.com/user-attachments/assets/9b478403-8045-4030-82c2-0c48397e366c




## Prerequisites

### 1. Install Ollama

First, install Ollama on your system:

**Windows:**
```bash
# Download from https://ollama.ai/download
# Or use winget
winget install Ollama.Ollama
```

**macOS:**
```bash
# Download from https://ollama.ai/download
# Or use Homebrew
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull Required Models

After installing Ollama, pull the required models:

```bash
# Pull the Qwen 2.5 3B model for both embeddings and chat
ollama pull qwen2.5:3b
```

### 3. Start Ollama Server

```bash
ollama serve
```

The server will run on `http://localhost:11434` by default.

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## Usage

### 1. Upload Documents

1. Navigate to the "Upload Documents" mode in the sidebar
2. Upload PDF or TXT files using the file uploader
3. Click "Process and Embed Files" to extract text, create chunks, and generate embeddings
4. Monitor the progress as files are processed

### 2. Chat with Documents

1. Switch to "Chat" mode
2. Configure RAG settings:
   - **Use RAG**: Toggle between RAG and direct chat
   - **Top K Results**: Number of similar chunks to retrieve
   - **Similarity Threshold**: Minimum similarity score for retrieved chunks
3. Customize the system prompt if desired
4. Start chatting! The assistant will use retrieved context when RAG is enabled

### 3. Database Management

1. Go to "Database Management" mode
2. View statistics about uploaded documents and chunks
3. Browse uploaded documents and delete them if needed
4. Review chat history
5. Export data or clear all data (functionality to be implemented)

## Project Structure

```
llm cuda/
├── app.py                 # Main Streamlit application
├── database.py           # SQLite database operations
├── file_processor.py     # Text extraction & chunking
├── embeddings.py         # Embedding calls & storage
├── retriever.py          # Query embedding & similarity search
├── chat.py              # Chat calls
├── requirements.txt     # Python dependencies
├── test_setup.py       # Setup verification script
└── README.md           # This file
```

## Database Schema

The application uses a SQLite database with the following tables:

- **documents**: Stores uploaded file metadata
  - `doc_id` (PRIMARY KEY)
  - `file_name`
  - `uploaded_at`

- **chunks**: Stores text chunks with embeddings
  - `chunk_id` (PRIMARY KEY)
  - `doc_id` (FOREIGN KEY)
  - `text`
  - `embedding` (BLOB)

- **chats**: Stores chat interactions
  - `chat_id` (PRIMARY KEY)
  - `user_input`
  - `bot_output`
  - `timestamp`

## Configuration

### Model Configuration

You can modify the model names in the respective manager classes:

- **EmbeddingManager** (`embeddings.py`): Change `model_name` parameter
- **ChatManager** (`chat.py`): Change `model_name` parameter

### Ollama Server URL

If your Ollama server is running on a different URL, update the `base_url` parameter in both `EmbeddingManager` and `ChatManager`.

### Chunking Parameters

Modify chunking behavior in `FileProcessor`:
- `chunk_size`: Number of characters per chunk (default: 500)
- `chunk_overlap`: Number of characters to overlap between chunks (default: 50)

## Troubleshooting

### Connection Issues

1. **Ollama server not running**: Make sure Ollama is installed and running with `ollama serve`
2. **Model not found**: Pull the required models with `ollama pull qwen2.5:3b`
3. **Port conflicts**: Check if port 11434 is available for Ollama

### Performance Issues

1. **Slow embeddings**: Reduce batch size in `EmbeddingManager.get_embeddings_batch()`
2. **Memory issues**: Process files one at a time or reduce chunk size
3. **Large files**: Consider splitting very large documents before upload

### Common Errors

1. **"Model not found"**: Pull the required Ollama models
2. **"Connection refused"**: Start the Ollama server
3. **"No text content found"**: Check if the PDF contains extractable text

## Development

### Adding New File Types

To support additional file types:

1. Add the file extension to the `extract_text()` method in `FileProcessor`
2. Implement the corresponding extraction method
3. Update the Streamlit file uploader to accept the new type

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests. 
