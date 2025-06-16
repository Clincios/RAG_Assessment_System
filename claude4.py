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
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
import base64
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

class PDFExporter:
    """Handles PDF export functionality for chat conversations"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for PDF export"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=HexColor('#2E86AB'),
            alignment=1  # Center alignment
        )
        
        # User message style
        self.user_style = ParagraphStyle(
            'UserMessage',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            spaceBefore=8,
            leftIndent=20,
            backgroundColor=HexColor('#E3F2FD'),
            borderPadding=8,
            borderWidth=1,
            borderColor=HexColor('#BBDEFB')
        )
        
        # Assistant message style
        self.assistant_style = ParagraphStyle(
            'AssistantMessage',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            spaceBefore=8,
            leftIndent=20,
            backgroundColor=HexColor('#F1F8E9'),
            borderPadding=8,
            borderWidth=1,
            borderColor=HexColor('#C8E6C9')
        )
        
        # Metadata style
        self.metadata_style = ParagraphStyle(
            'Metadata',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=HexColor('#666666'),
            spaceAfter=6
        )
        
        # Header style for sections
        self.header_style = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=HexColor('#1976D2')
        )
    
    def clean_text_for_pdf(self, text: str) -> str:
        """Clean text for PDF export by handling special characters"""
        # Replace problematic characters
        replacements = {
            '•': '*',
            '–': '-',
            '—': '-',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '…': '...',
            '\u2022': '*',  # bullet point
            '\u2013': '-',  # en dash
            '\u2014': '-',  # em dash
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove or replace other problematic unicode characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text
    
    def create_chat_pdf(self, chat_history: List[Dict], document_name: str = None) -> bytes:
        """Create PDF from chat history"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build PDF content
        story = []
        
        # Title
        title_text = f"Chat Conversation Export"
        if document_name:
            title_text += f" - {document_name}"
        story.append(Paragraph(title_text, self.title_style))
        story.append(Spacer(1, 20))
        
        # Metadata
        export_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_text = f"Exported on: {export_time}"
        if document_name:
            metadata_text += f"<br/>Source Document: {document_name}"
        metadata_text += f"<br/>Total Messages: {len(chat_history)}"
        
        story.append(Paragraph(metadata_text, self.metadata_style))
        story.append(Spacer(1, 20))
        
        # Chat messages
        if chat_history:
            story.append(Paragraph("Conversation History", self.header_style))
            
            for i, message in enumerate(chat_history, 1):
                role = message.get('role', 'unknown')
                content = message.get('message', '')
                
                # Clean the content for PDF
                content = self.clean_text_for_pdf(content)
                
                # Add message number and role
                if role == 'user':
                    prefix = f"<b>User (Message {i}):</b><br/>"
                    style = self.user_style
                elif role == 'assistant':
                    prefix = f"<b>Assistant (Message {i}):</b><br/>"
                    style = self.assistant_style
                else:
                    prefix = f"<b>{role.title()} (Message {i}):</b><br/>"
                    style = self.styles['Normal']
                
                # Handle long messages by splitting them
                max_length = 2000  # Adjust based on needs
                if len(content) > max_length:
                    # Split long content into chunks
                    chunks = [content[i:i+max_length] for i in range(0, len(content), max_length)]
                    for j, chunk in enumerate(chunks):
                        if j == 0:
                            story.append(Paragraph(prefix + chunk, style))
                        else:
                            story.append(Paragraph(f"<i>(continued...)</i><br/>{chunk}", style))
                        story.append(Spacer(1, 6))
                else:
                    story.append(Paragraph(prefix + content, style))
                    story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("No conversation history available.", self.styles['Normal']))
        
        # Build PDF
        try:
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error creating PDF: {str(e)}")
            # Create a simple error PDF
            error_buffer = io.BytesIO()
            error_doc = SimpleDocTemplate(error_buffer, pagesize=A4)
            error_story = [
                Paragraph("PDF Export Error", self.title_style),
                Spacer(1, 20),
                Paragraph(f"An error occurred while creating the PDF: {str(e)}", self.styles['Normal']),
                Spacer(1, 12),
                Paragraph("Please try again or contact support if the issue persists.", self.styles['Normal'])
            ]
            error_doc.build(error_story)
            error_buffer.seek(0)
            return error_buffer.getvalue()
    
    def create_summary_pdf(self, summary_content: str, document_name: str = None) -> bytes:
        """Create PDF from summary content"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Title
        title = "Document Summary"
        if document_name:
            title += f" - {document_name}"
        story.append(Paragraph(title, self.title_style))
        story.append(Spacer(1, 20))
        
        # Metadata
        export_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_text = f"Generated on: {export_time}"
        if document_name:
            metadata_text += f"<br/>Source Document: {document_name}"
        
        story.append(Paragraph(metadata_text, self.metadata_style))
        story.append(Spacer(1, 20))
        
        # Summary content
        cleaned_content = self.clean_text_for_pdf(summary_content)
        story.append(Paragraph("Summary", self.header_style))
        story.append(Paragraph(cleaned_content, self.styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

class PDFChatbotConfig:
    """Configuration class for the chatbot"""
    def __init__(self):
        # Get API key from environment variable
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not self.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Please create a .env file with your API key or set it in your environment."
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
        self.pdf_exporter = PDFExporter()
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
    
    def download_chat_as_pdf(self):
        """Create download button for chat history as PDF"""
        if st.session_state.chat_history:
            try:
                pdf_bytes = self.pdf_exporter.create_chat_pdf(
                    st.session_state.chat_history,
                    st.session_state.current_document
                )
                
                # Create filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chat_export_{timestamp}.pdf"
                if st.session_state.current_document:
                    doc_name = st.session_state.current_document.replace('.pdf', '')
                    filename = f"chat_{doc_name}_{timestamp}.pdf"
                
                st.download_button(
                    label="📥 Download Chat as PDF",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True,
                    help="Download the entire conversation as a formatted PDF document"
                )
                
            except Exception as e:
                st.error(f"Error creating PDF: {str(e)}")
                logger.error(f"PDF creation error: {str(e)}")
        else:
            st.info("No chat history to download")
    
    def download_content_as_pdf(self, content: str, content_type: str):
        """Create download button for specific content as PDF"""
        try:
            pdf_bytes = self.pdf_exporter.create_summary_pdf(
                content,
                st.session_state.current_document
            )
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{content_type.lower().replace(' ', '_')}_{timestamp}.pdf"
            if st.session_state.current_document:
                doc_name = st.session_state.current_document.replace('.pdf', '')
                filename = f"{content_type.lower().replace(' ', '_')}_{doc_name}_{timestamp}.pdf"
            
            st.download_button(
                label=f"📥 Download {content_type} as PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True,
                help=f"Download the {content_type.lower()} as a formatted PDF document"
            )
            
        except Exception as e:
            st.error(f"Error creating {content_type} PDF: {str(e)}")
            logger.error(f"{content_type} PDF creation error: {str(e)}")
    
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
                    message_placeholder.markdown(response["result"])
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "message": response["result"]
                    })
                    
                    # Add download button for the response
                    self.download_content_as_pdf(response["result"], title)
                    
            except Exception as e:
                error_msg = f"Error generating {title.lower()}: {str(e)}"
                message_placeholder.error(error_msg)
                logger.error(error_msg)

def main():
    """Main function to run the Streamlit app"""
    # Page configuration
    st.set_page_config(
        page_title="PDF Document Chatbot",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize the chatbot UI
    chatbot = ChatbotUI()
    
    # Header
    st.title("📄 PDF Document Chatbot")
    st.markdown("Upload a PDF document and ask questions about its content!")
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help=f"Upload a PDF file (max {chatbot.config.MAX_FILE_SIZE_MB}MB)"
        )
        
        if uploaded_file is not None:
            if chatbot.validate_uploaded_file(uploaded_file):
                # Process the file if it's new or different
                if (st.session_state.current_document != uploaded_file.name):
                    chatbot.process_uploaded_file(uploaded_file)
                
                # Show current document info
                st.success(f"📄 **Current Document:** {st.session_state.current_document}")
                
                # Document statistics
                if st.session_state.vectorstore:
                    try:
                        # Get document count from vectorstore
                        collection = st.session_state.vectorstore._collection
                        doc_count = collection.count()
                        st.info(f"📊 **Document chunks:** {doc_count}")
                    except:
                        st.info("📊 **Status:** Document processed and ready")
        
        # Chat controls
        if st.session_state.chat_history:
            st.header("Chat Controls")
            
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.memory.clear()
                st.rerun()
            
            # Download chat as PDF
            chatbot.download_chat_as_pdf()
        
        # Settings
        st.header("Settings")
        
        # Temperature control
        temperature = st.slider(
            "Response Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Lower values make responses more focused and deterministic"
        )
        
        if temperature != st.session_state.llm.temperature:
            st.session_state.llm.temperature = temperature
        
        # Memory length control
        memory_length = st.slider(
            "Chat Memory Length",
            min_value=5,
            max_value=20,
            value=chatbot.config.MAX_MEMORY_MESSAGES,
            help="Number of previous messages to remember"
        )
        
        if memory_length != st.session_state.memory.k:
            st.session_state.memory.k = memory_length
    
    # Main content area
    if st.session_state.vectorstore is not None:
        # Quick actions
        chatbot.display_quick_actions()
        
        # Chat interface
        st.subheader("💬 Chat with your document")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["message"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "message": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    with st.spinner("Thinking..."):
                        # Get response from QA chain
                        response = st.session_state.qa_chain.invoke({"query": prompt})
                        
                        # Display response
                        message_placeholder.markdown(response["result"])
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "message": response["result"]
                        })
                
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    message_placeholder.error(error_msg)
                    logger.error(error_msg)
    
    else:
        # Welcome screen when no document is loaded
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>👋 Welcome to PDF Document Chatbot!</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Upload a PDF document using the sidebar to get started.
            </p>
            <p>
                Once uploaded, you can:
            </p>
            <ul style="text-align: left; display: inline-block;">
                <li>📋 Get automatic document summaries</li>
                <li>📚 Generate study guides</li>
                <li>❓ Create exam questions</li>
                <li>💬 Ask specific questions about the content</li>
                <li>📥 Export conversations as PDF</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample questions
        st.subheader("💡 Example Questions You Can Ask:")
        sample_questions = [
            "What is the main topic of this document?",
            "Can you summarize the key findings?",
            "What are the most important points to remember?",
            "Are there any specific recommendations mentioned?",
            "What data or statistics are presented?"
        ]
        
        for question in sample_questions:
            st.markdown(f"• {question}")

if __name__ == "__main__":
    main()