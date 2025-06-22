import streamlit as st
import os
import time
from typing import List, Tuple
import tempfile

# Import our custom modules
from database import DatabaseManager
from file_processor import FileProcessor
from embeddings import EmbeddingManager
from retriever import Retriever
from chat import ChatManager

# Page configuration
st.set_page_config(
    page_title="Document Chat with RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

if 'file_processor' not in st.session_state:
    st.session_state.file_processor = FileProcessor()

if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = EmbeddingManager()

if 'retriever' not in st.session_state:
    st.session_state.retriever = Retriever(
        st.session_state.db_manager, 
        st.session_state.embedding_manager
    )

if 'chat_manager' not in st.session_state:
    st.session_state.chat_manager = ChatManager()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
st.sidebar.title("📚 Document Chat with RAG")
st.sidebar.markdown("---")

# Mode selection
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Upload Documents", "Chat", "Database Management"],
    index=1
)

# Connection status
st.sidebar.markdown("### 🔗 Connection Status")
embedding_status = st.session_state.embedding_manager.test_connection()
chat_status = st.session_state.chat_manager.test_connection()

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Embedding", "✅" if embedding_status else "❌")
with col2:
    st.metric("Chat", "✅" if chat_status else "❌")

# Main content
if mode == "Upload Documents":
    st.title("📤 Upload Documents")
    st.markdown("Upload PDF and TXT files to create embeddings for RAG.")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF or TXT files to process and embed"
    )
    
    if uploaded_files:
        st.markdown("### 📋 Uploaded Files")
        
        # Display file information
        for file in uploaded_files:
            file_info = st.session_state.file_processor.get_file_info(file)
            st.write(f"**{file_info['name']}** ({file_info['size']} bytes)")
        
        # Process button
        if st.button("🚀 Process and Embed Files", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_files = len(uploaded_files)
            processed_files = 0
            
            for file in uploaded_files:
                try:
                    status_text.text(f"Processing {file.name}...")
                    
                    # Process file
                    text, chunks = st.session_state.file_processor.process_file(file)
                    
                    if not chunks:
                        st.warning(f"No text content found in {file.name}")
                        continue
                    
                    # Add document to database
                    doc_id = st.session_state.db_manager.add_document(file.name)
                    
                    # Get embeddings for chunks
                    status_text.text(f"Generating embeddings for {file.name}...")
                    embeddings = st.session_state.embedding_manager.get_embeddings_batch(chunks)
                    
                    # Store chunks and embeddings
                    for chunk, embedding in zip(chunks, embeddings):
                        st.session_state.db_manager.add_chunk(doc_id, chunk, embedding)
                    
                    processed_files += 1
                    progress_bar.progress(processed_files / total_files)
                    
                    st.success(f"✅ Successfully processed {file.name} ({len(chunks)} chunks)")
                    
                except Exception as e:
                    st.error(f"❌ Error processing {file.name}: {str(e)}")
            
            status_text.text("🎉 Processing complete!")
            time.sleep(2)
            st.rerun()

elif mode == "Chat":
    st.title("💬 Chat with Documents")
    
    # Chat settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_rag = st.checkbox("Use RAG", value=True, help="Enable retrieval-augmented generation")
    
    with col2:
        top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5, help="Number of similar chunks to retrieve")
    
    with col3:
        similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.1, help="Minimum similarity score for retrieved chunks")
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.chat_manager.get_default_system_prompt(),
        height=100,
        help="Customize the assistant's behavior"
    )
    
    st.markdown("---")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                if use_rag:
                    # Retrieve relevant chunks
                    retrieved_chunks = st.session_state.retriever.retrieve_similar_chunks(
                        prompt, top_k, similarity_threshold
                    )
                    
                    if retrieved_chunks:
                        st.info(f"📚 Retrieved {len(retrieved_chunks)} relevant chunks")
                        
                        # Show retrieved chunks in expander
                        with st.expander("🔍 Retrieved Context"):
                            for i, (chunk_id, text, score) in enumerate(retrieved_chunks):
                                st.markdown(f"**Chunk {i+1}** (Score: {score:.3f})")
                                st.text(text[:200] + "..." if len(text) > 200 else text)
                                st.markdown("---")
                    
                    # Stream response with RAG
                    for chunk in st.session_state.chat_manager.chat_with_rag_stream(
                        prompt, retrieved_chunks, system_prompt
                    ):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                else:
                    # Direct chat without RAG
                    for chunk in st.session_state.chat_manager.chat_with_stream(
                        prompt, system_prompt=system_prompt
                    ):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                
                # Final response without cursor
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Save to database
                st.session_state.db_manager.add_chat(prompt, full_response)
                st.session_state.chat_manager.add_to_history(prompt, full_response)
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_manager.clear_history()
        st.rerun()

elif mode == "Database Management":
    st.title("🗄️ Database Management")
    
    # Statistics
    stats = st.session_state.retriever.get_document_statistics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", stats['total_documents'])
    with col2:
        st.metric("Total Chunks", stats['total_chunks'])
    with col3:
        st.metric("Chat History", len(st.session_state.db_manager.get_chat_history()))
    
    st.markdown("---")
    
    # Document list
    st.markdown("### 📄 Uploaded Documents")
    documents = st.session_state.db_manager.get_documents()
    
    if documents:
        for doc_id, file_name, uploaded_at in documents:
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.write(f"**{file_name}**")
            with col2:
                st.write(f"Uploaded: {uploaded_at}")
            with col3:
                if st.button(f"Delete", key=f"del_{doc_id}"):
                    st.session_state.db_manager.delete_document(doc_id)
                    st.success(f"Deleted {file_name}")
                    st.rerun()
    else:
        st.info("No documents uploaded yet.")
    
    st.markdown("---")
    
    # Chat history
    st.markdown("### 💬 Recent Chat History")
    chat_history = st.session_state.db_manager.get_chat_history(limit=10)
    
    if chat_history:
        for user_input, bot_output, timestamp in chat_history:
            with st.expander(f"💬 {user_input[:50]}..."):
                st.markdown(f"**User:** {user_input}")
                st.markdown(f"**Assistant:** {bot_output}")
                st.caption(f"Timestamp: {timestamp}")
    else:
        st.info("No chat history yet.")
    
    # Export/Import options
    st.markdown("---")
    st.markdown("### 📤 Export Options")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Chat History"):
            # This would implement export functionality
            st.info("Export functionality would be implemented here")
    
    with col2:
        if st.button("🗑️ Clear All Data"):
            if st.checkbox("I understand this will delete all data"):
                # This would implement clear functionality
                st.warning("Clear functionality would be implemented here")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • Powered by Ollama • Document Chat with RAG</p>
    </div>
    """,
    unsafe_allow_html=True
) 