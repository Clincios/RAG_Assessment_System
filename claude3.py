import streamlit as st
import os
import time
import hashlib
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Optional
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler

# Custom callback handler for streaming
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

class PDFChatbotConfig:
    """Configuration class for the chatbot"""
    def __init__(self):
        # Get API key from environment variable
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not self.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please create a .env file with your API key or set it as an environment variable."
            )
        
        self.PDF_DIR = Path("pdfFiles")
        self.VECTOR_DB_DIR = Path("vectorDB")
        self.METADATA_FILE = Path("pdf_metadata.json")
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        self.MAX_MEMORY_MESSAGES = 10
        self.SUPPORTED_FILE_TYPES = ["pdf"]
        self.MAX_FILE_SIZE_MB = 50

class PDFProcessor:
    """Handles PDF processing and vectorization"""
    
    def __init__(self, config: PDFChatbotConfig):
        self.config = config
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=config.GOOGLE_API_KEY
        )
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file to check if it's already processed"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            logger.error(f"File not found when generating hash: {file_path}")
            raise
    
    def get_document_specific_db_path(self, file_hash: str) -> Path:
        """Get document-specific vector database path"""
        return self.config.VECTOR_DB_DIR / f"doc_{file_hash}"
    
    def load_metadata(self) -> Dict:
        """Load metadata about processed files"""
        if self.config.METADATA_FILE.exists():
            try:
                with open(self.config.METADATA_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Error loading metadata: {e}. Creating new metadata file.")
                return {}
        return {}
    
    def save_metadata(self, metadata: Dict):
        """Save metadata about processed files"""
        try:
            with open(self.config.METADATA_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def is_file_processed(self, file_path: Path) -> tuple[bool, str]:
        """Check if file has already been processed and return file hash"""
        try:
            if not file_path.exists():
                return False, ""
            
            file_hash = self.get_file_hash(file_path)
            metadata = self.load_metadata()
            doc_db_path = self.get_document_specific_db_path(file_hash)
            
            # Check if metadata exists and vector db directory exists
            is_processed = (file_hash in metadata and 
                          doc_db_path.exists() and 
                          any(doc_db_path.iterdir()))
            
            return is_processed, file_hash
        except Exception as e:
            logger.error(f"Error checking if file is processed: {e}")
            return False, ""
    
    def process_pdf(self, file_path: Path) -> List:
        """Process PDF and return document chunks"""
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            # Convert Path to string for PyPDFLoader
            loader = PyPDFLoader(str(file_path.resolve()))
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content found in PDF")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            
            # Add metadata to chunks
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'source_file': file_path.name,
                    'chunk_index': i,
                    'processed_at': datetime.now().isoformat()
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
    
    def create_vectorstore(self, documents: List, file_hash: str) -> Chroma:
        """Create document-specific vector store"""
        try:
            # Get document-specific database path
            doc_db_path = self.get_document_specific_db_path(file_hash)
            doc_db_path.mkdir(parents=True, exist_ok=True)
            
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(doc_db_path)
            )
            vectorstore.persist()
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vectorstore(self, file_hash: str) -> Optional[Chroma]:
        """Load document-specific vector store"""
        try:
            doc_db_path = self.get_document_specific_db_path(file_hash)
            
            if doc_db_path.exists() and any(doc_db_path.iterdir()):
                vectorstore = Chroma(
                    persist_directory=str(doc_db_path),
                    embedding_function=self.embeddings
                )
                logger.info(f"Vectorstore loaded for document hash: {file_hash}")
                return vectorstore
        except Exception as e:
            logger.warning(f"Could not load vectorstore for {file_hash}: {e}")
        return None
    
    def cleanup_old_vectorstores(self, keep_hash: str = None):
        """Clean up old vector stores (optional - for storage management)"""
        try:
            if not self.config.VECTOR_DB_DIR.exists():
                return
                
            for item in self.config.VECTOR_DB_DIR.iterdir():
                if item.is_dir() and item.name.startswith("doc_"):
                    if keep_hash is None or not item.name.endswith(keep_hash):
                        shutil.rmtree(item)
                        logger.info(f"Cleaned up old vectorstore: {item}")
        except Exception as e:
            logger.warning(f"Error cleaning up old vectorstores: {e}")

class ChatbotUI:
    """Handles the Streamlit UI"""
    
    def __init__(self):
        self.config = PDFChatbotConfig()
        self.processor = PDFProcessor(self.config)
        self.setup_directories()
        self.initialize_session_state()
    
    def setup_directories(self):
        """Create necessary directories"""
        for directory in [self.config.PDF_DIR, self.config.VECTOR_DB_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created/verified: {directory}")
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        
        # Enhanced prompt template
        if 'template' not in st.session_state:
            st.session_state.template = """You are an intelligent document assistant specialized in analyzing and answering questions about PDF documents. 
            You provide accurate, detailed, and helpful responses based on the provided context.

            Instructions:
            - Answer questions directly and accurately based on the context
            - If information is not available in the context, clearly state that
            - Provide specific page references when possible
            - Use professional and informative tone
            - For complex queries, break down your response into clear sections
            - IMPORTANT: Only use information from the currently active document context

            Context: {context}
            Chat History: {history}

            Question: {question}
            Assistant:"""
        
        if 'prompt' not in st.session_state:
            st.session_state.prompt = PromptTemplate(
                input_variables=["history", "context", "question"],
                template=st.session_state.template,
            )
        
        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferWindowMemory(
                memory_key="history",
                return_messages=True,
                input_key="question",
                k=self.config.MAX_MEMORY_MESSAGES
            )
        
        if 'llm' not in st.session_state:
            st.session_state.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.config.GOOGLE_API_KEY,
                temperature=0.1,
                max_output_tokens=4096,
                streaming=True
            )
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None
        
        if 'current_document' not in st.session_state:
            st.session_state.current_document = None
        
        if 'current_document_hash' not in st.session_state:
            st.session_state.current_document_hash = None
        
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = None
    
    def reset_document_session(self):
        """Reset session for new document"""
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.session_state.vectorstore = None
        st.session_state.qa_chain = None
        st.session_state.current_document = None
        st.session_state.current_document_hash = None
        logger.info("Document session reset")
    
    def validate_uploaded_file(self, uploaded_file) -> bool:
        """Validate uploaded file"""
        if uploaded_file is None:
            return False
        
        # Check file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in self.config.SUPPORTED_FILE_TYPES:
            st.error(f"Unsupported file type. Please upload: {', '.join(self.config.SUPPORTED_FILE_TYPES)}")
            return False
        
        # Check file size
        if uploaded_file.size > self.config.MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File too large. Maximum size: {self.config.MAX_FILE_SIZE_MB}MB")
            return False
        
        return True
    
    def save_uploaded_file(self, uploaded_file) -> Path:
        """Save uploaded file to disk"""
        self.config.PDF_DIR.mkdir(parents=True, exist_ok=True)
        file_path = self.config.PDF_DIR / uploaded_file.name
        
        try:
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())
            logger.info(f"File saved successfully: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise
    
    def process_uploaded_file(self, uploaded_file):
        """Process the uploaded PDF file"""
        try:
            # Save the uploaded file
            file_path = self.save_uploaded_file(uploaded_file)
            logger.info(f"Processing file: {file_path}")
            
            # Check if this is a different document than currently loaded
            is_processed, file_hash = self.processor.is_file_processed(file_path)
            
            # If switching to a different document, reset session
            if (st.session_state.current_document_hash and 
                st.session_state.current_document_hash != file_hash):
                st.info("Switching to a different document. Resetting chat session...")
                self.reset_document_session()
            
            # Set current document info
            st.session_state.current_document = uploaded_file.name
            st.session_state.current_document_hash = file_hash
            
            if is_processed:
                st.info("Document already processed. Loading existing analysis...")
                st.session_state.vectorstore = self.processor.load_vectorstore(file_hash)
                if st.session_state.vectorstore is None:
                    st.warning("Could not load existing analysis. Reprocessing document...")
                    self._process_new_file(file_path, uploaded_file, file_hash)
            else:
                self._process_new_file(file_path, uploaded_file, file_hash)
            
            # Create QA chain
            self.create_qa_chain()
            
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            st.error(error_msg)
            logger.error(error_msg)
            
            # Clean up if file was partially saved
            file_path = self.config.PDF_DIR / uploaded_file.name
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up partial file: {file_path}")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up file: {cleanup_error}")
    
    def _process_new_file(self, file_path: Path, uploaded_file, file_hash: str):
        """Process a new PDF file"""
        with st.status("Processing PDF...", expanded=True) as status:
            st.write("📄 Reading PDF content...")
            
            # Process PDF
            st.write("🔄 Splitting document into chunks...")
            documents = self.processor.process_pdf(file_path)
            
            st.write("🧮 Creating embeddings and vector store...")
            st.session_state.vectorstore = self.processor.create_vectorstore(documents, file_hash)
            
            # Update metadata
            metadata = self.processor.load_metadata()
            metadata[file_hash] = {
                'filename': uploaded_file.name,
                'processed_at': datetime.now().isoformat(),
                'chunks_count': len(documents)
            }
            self.processor.save_metadata(metadata)
            
            status.update(label="✅ PDF processed successfully!", state="complete")
    
    def create_qa_chain(self):
        """Create the QA chain"""
        if st.session_state.vectorstore is not None:
            try:
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 5,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )
                
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=st.session_state.llm,
                    chain_type='stuff',
                    retriever=retriever,
                    verbose=False,
                    chain_type_kwargs={
                        "prompt": st.session_state.prompt,
                        "memory": st.session_state.memory,
                    }
                )
                logger.info("QA chain created successfully")
            except Exception as e:
                logger.error(f"Error creating QA chain: {e}")
                st.error("Error setting up question-answering system. Please try again.")
    
    def generate_automated_response(self, feature_type: str):
        """Generate automated responses for quick features"""
        prompts = {
            "summarize": f"""Please provide a comprehensive summary of the current document: "{st.session_state.current_document}". Include:
            - Main topic and purpose
            - Key points and findings
            - Important conclusions or recommendations
            - Any significant data or statistics mentioned
            Keep the summary concise but thorough. Focus only on this document.""",
            
            "study_guide": f"""Create a detailed study guide from the current document: "{st.session_state.current_document}" that includes:
            - Main concepts and definitions
            - Important facts and figures
            - Key processes or procedures explained
            - Critical points students should remember
            - Potential areas of focus for deeper study
            Format it as a structured study guide. Use only information from this document.""",
            
            "exam_questions": f"""Generate a set of potential examination questions based on the current document: "{st.session_state.current_document}". Include:
            - 3-5 multiple choice questions with options
            - 3-5 short answer questions
            - 2-3 essay/long answer questions
            - Include the correct answers or key points for each question
            Focus on testing understanding of key concepts from this specific document."""
        }
        
        return prompts.get(feature_type, "")
    
    def display_quick_actions(self):
        """Display quick action buttons for automated features"""
        if st.session_state.vectorstore is not None:
            st.subheader("🚀 Quick Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Summarize Document", use_container_width=True):
                    prompt = self.generate_automated_response("summarize")
                    self.handle_automated_request(prompt, "Document Summary")
            
            with col2:
                if st.button("📚 Generate Study Guide", use_container_width=True):
                    prompt = self.generate_automated_response("study_guide")
                    self.handle_automated_request(prompt, "Study Guide")
            
            with col3:
                if st.button("❓ Exam Questions", use_container_width=True):
                    prompt = self.generate_automated_response("exam_questions")
                    self.handle_automated_request(prompt, "Examination Questions")
            
            st.divider()
    
    def handle_automated_request(self, prompt: str, title: str):
        """Handle automated feature requests"""
        # Add automated request to chat history
        automated_message = {"role": "user", "message": f"🤖 {title}"}
        st.session_state.chat_history.append(automated_message)
        
        # Display the automated request
        with st.chat_message("user"):
            st.markdown(f"🤖 **{title}**")
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                with st.spinner(f"Generating {title.lower()}..."):
                    response = st.session_state.qa_chain.invoke({"query": prompt})
                
                # Display response with typing effect
                full_response = response['result']
                displayed_response = ""
                
                for chunk in full_response.split():
                    displayed_response += chunk + " "
                    time.sleep(0.02)
                    message_placeholder.markdown(displayed_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add response to chat history
                assistant_message = {"role": "assistant", "message": full_response}
                st.session_state.chat_history.append(assistant_message)
                
            except Exception as e:
                error_message = f"Sorry, I couldn't generate the {title.lower()}: {str(e)}"
                message_placeholder.markdown(error_message)
                st.error("Please try again or check your document.")
                logger.error(f"Automated request error: {str(e)}")
        
        st.rerun()
    
    def handle_user_input(self, user_input: str):
        """Handle user input and generate response"""
        # Add user message to chat history
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                with st.spinner("🤔 Thinking..."):
                    response = st.session_state.qa_chain.invoke({"query": user_input})
                
                # Simulate streaming response
                full_response = response['result']
                displayed_response = ""
                
                for chunk in full_response.split():
                    displayed_response += chunk + " "
                    time.sleep(0.03)
                    message_placeholder.markdown(displayed_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add assistant message to chat history
                assistant_message = {"role": "assistant", "message": full_response}
                st.session_state.chat_history.append(assistant_message)
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                message_placeholder.markdown(error_message)
                st.error("Please try again or upload a different PDF.")
                logger.error(f"Query processing error: {str(e)}")
    
    def display_sidebar(self):
        """Display sidebar with additional information"""
        with st.sidebar:
            st.header("📊 Document Manager")
            
            if st.session_state.current_document:
                st.subheader("📄 Active Document:")
                st.success(f"📝 {st.session_state.current_document}")
                st.caption(f"Hash: {st.session_state.current_document_hash[:8]}...")
            else:
                st.info("No document loaded")
            
            st.divider()
            
            st.subheader("🛠️ Tools")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.session_state.memory.clear()
                    st.rerun()
            
            with col2:
                if st.button("🔄 Reset App", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            
            # Optional: Add cleanup button for storage management
            if st.button("🧹 Cleanup Old Files", use_container_width=True):
                try:
                    current_hash = st.session_state.current_document_hash
                    self.processor.cleanup_old_vectorstores(current_hash)
                    st.success("Cleaned up old vector stores")
                except Exception as e:
                    st.error(f"Cleanup failed: {str(e)}")
            
            st.divider()
            
            st.subheader("ℹ️ Features")
            features = [
                "📖 Document Isolation",
                "🧠 Smart Chunking", 
                "💭 Memory Context",
                "🎯 Source Citations",
                "⚡ File Caching",
                "🔐 Session Management"
            ]
            for feature in features:
                st.write(feature)

    def display_chat_history(self):
        """Display the chat history in the Streamlit interface"""
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["message"])

    def display_welcome_message(self):
        """Display welcome message and features when no PDF is loaded"""
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h2>Welcome to PDF Chatbot Pro! 📚</h2>
            <p style='font-size: 1.2rem; color: #666;'>
                Your AI-powered document analysis assistant with isolated document sessions
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### ✨ Key Features:
        - **Document Isolation**: Each PDF gets its own isolated analysis session
        - **Smart Document Analysis**: Upload any PDF and get instant insights
        - **Interactive Chat**: Ask questions about your document in natural language
        - **Quick Actions**: Generate summaries, study guides, and exam questions
        - **Memory Context**: Maintains conversation context for the current document
        - **Source Citations**: Get accurate references to document content
        - **Session Management**: Automatic session reset when switching documents
        
        ### 🚀 Getting Started:
        1. Upload your PDF document using the uploader above
        2. Wait for the document to be processed
        3. Start asking questions or use quick actions
        4. Upload a different PDF to start a new isolated session
        """)
    
    def run(self):
        """Run the main application"""
        st.set_page_config(
            page_title="PDF Chatbot Pro",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better UI
        st.markdown("""
        <style>
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .success-message {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("📚 PDF Chatbot Pro")
        st.markdown("*Unlock the power of your documents with AI-driven analysis and isolated sessions*")
        
        # Display sidebar
        self.display_sidebar()
        
        # File upload section
        st.subheader("📤 Upload Your Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file to analyze",
            type=self.config.SUPPORTED_FILE_TYPES,
            help=f"Maximum file size: {self.config.MAX_FILE_SIZE_MB}MB. Each document gets an isolated session."
        )
        
        # Process uploaded file
        if uploaded_file is not None and self.validate_uploaded_file(uploaded_file):
            # Check if this is the currently active document
            if st.session_state.current_document != uploaded_file.name:
                self.process_uploaded_file(uploaded_file)
        
        # Main content area
        if st.session_state.vectorstore is not None and st.session_state.current_document:
            # Success message
            st.markdown(f"""
            <div class="success-message">
                <strong>✅ Document "{st.session_state.current_document}" loaded successfully!</strong> 
                You can now use quick actions or ask specific questions about this document.
            </div>
            """, unsafe_allow_html=True)
            
            # Display quick actions
            self.display_quick_actions()
            
            # Chat interface
            st.subheader("💬 Chat with Your Document")
            
            # Display chat history
            self.display_chat_history()
            
            # Chat input
            if user_input := st.chat_input("Ask any question about your document..."):
                self.handle_user_input(user_input)
        
        else:
            # Welcome message when no PDF is loaded
            self.display_welcome_message()
            
            st.info("👆 Upload a PDF document above to start using all features!")
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>Built with ❤️ by -Clinton Ageboba- | Document Isolation Enabled</div>", 
            unsafe_allow_html=True
        )

# Run the application
if __name__ == "__main__":
    app = ChatbotUI()
    app.run()