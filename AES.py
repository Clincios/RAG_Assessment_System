import streamlit as st
import os
import time
import hashlib
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
import shutil
import re
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Load environment variables
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
import chromadb
from chromadb.config import Settings

# Custom callback handler for streaming
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

@dataclass
class ScoringResult:
    """Data class for storing scoring results"""
    question: str
    user_answer: str
    correct_answer: str
    score: float
    feedback: str
    max_score: float = 10.0

@dataclass
class AssessmentResult:
    """Data class for storing complete assessment results"""
    assessment_id: str
    document_name: str
    total_score: float
    max_total_score: float
    percentage: float
    question_results: List[ScoringResult]
    timestamp: str
    time_taken: Optional[str] = None

class PDFChatbotConfig:
    """Configuration class for the chatbot"""
    def __init__(self):
        # Get API key from environment variable
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")
            
        self.PDF_DIR = Path("pdfFiles")
        self.VECTOR_DB_DIR = Path("vectorDB")
        self.METADATA_FILE = Path("pdf_metadata.json")
        self.ASSESSMENTS_DIR = Path("assessments")
        self.SCORING_RESULTS_FILE = Path("scoring_results.json")
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        self.MAX_MEMORY_MESSAGES = 10
        self.SUPPORTED_FILE_TYPES = ["pdf"]
        self.MAX_FILE_SIZE_MB = 50

class ScoringEngine:
    """Handles automated scoring functionality"""
    
    def __init__(self, config: PDFChatbotConfig, llm):
        self.config = config
        self.llm = llm
        self.config.ASSESSMENTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Scoring prompt template
        self.scoring_prompt = PromptTemplate(
            input_variables=["question", "correct_answer", "user_answer", "context"],
            template="""You are an expert educational assessor. Your task is to score a student's answer based on the provided context from a document.

SCORING CRITERIA:
- Maximum score: 10 points
- Award points based on accuracy, completeness, and understanding
- Consider partial credit for partially correct answers
- Be fair but maintain academic standards

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}

CORRECT/EXPECTED ANSWER: {correct_answer}

STUDENT'S ANSWER: {user_answer}

Please provide your assessment in the following JSON format:
{{
    "score": [numeric score out of 10],
    "feedback": "[detailed feedback explaining the score, what was correct, what was missing, and suggestions for improvement]"
}}

Be constructive in your feedback and explain your reasoning clearly."""
        )
    
    def load_scoring_results(self) -> List[AssessmentResult]:
        """Load previous scoring results"""
        if self.config.SCORING_RESULTS_FILE.exists():
            try:
                with open(self.config.SCORING_RESULTS_FILE, 'r') as f:
                    data = json.load(f)
                    return [AssessmentResult(**result) for result in data]
            except Exception as e:
                logger.error(f"Error loading scoring results: {e}")
                return []
        return []
    
    def save_scoring_results(self, results: List[AssessmentResult]):
        """Save scoring results"""
        try:
            with open(self.config.SCORING_RESULTS_FILE, 'w') as f:
                json.dump([asdict(result) for result in results], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving scoring results: {e}")
    
    def generate_questions_for_assessment(self, vectorstore, num_questions: int = 5, difficulty: str = "medium") -> List[Dict]:
        """Generate questions for assessment based on document content"""
        try:
            difficulty_prompts = {
                "easy": "Create basic recall and understanding questions that test fundamental concepts.",
                "medium": "Create analytical questions that require understanding and application of concepts.",
                "hard": "Create complex questions that require critical thinking, analysis, and synthesis."
            }
            
            prompt = f"""Based on the provided document context, generate {num_questions} examination questions for assessment.

REQUIREMENTS:
- {difficulty_prompts.get(difficulty, difficulty_prompts["medium"])}
- Include the correct answer for each question
- Questions should be answerable based on the document content
- Vary question types (short answer, explanation, analysis)
- Each question should test different aspects of the material

Please format your response as a JSON array with this structure:
[
    {{
        "question": "Your question here",
        "correct_answer": "The correct/expected answer",
        "question_type": "short_answer|explanation|analysis",
        "difficulty": "{difficulty}"
    }}
]

Focus on the most important concepts and information from the document."""
            
            # Get relevant context from vectorstore
            retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
            docs = retriever.get_relevant_documents("key concepts main topics important information")
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate questions
            response = self.llm.invoke(f"Context: {context}\n\n{prompt}")
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
                if json_match:
                    questions_data = json.loads(json_match.group())
                    return questions_data
                else:
                    # Fallback: create questions manually if JSON parsing fails
                    return self._create_fallback_questions(context)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from LLM response, using fallback")
                return self._create_fallback_questions(context)
                
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return self._create_fallback_questions("Document content")
    
    def _create_fallback_questions(self, context: str) -> List[Dict]:
        """Create fallback questions if automatic generation fails"""
        return [
            {
                "question": "What are the main topics discussed in this document?",
                "correct_answer": "Summarize the key topics and themes presented in the document based on your reading.",
                "question_type": "explanation",
                "difficulty": "medium"
            },
            {
                "question": "Explain the most important concept presented in the document.",
                "correct_answer": "Identify and explain the central concept or principle discussed in the document.",
                "question_type": "explanation", 
                "difficulty": "medium"
            },
            {
                "question": "What conclusions or recommendations are made in the document?",
                "correct_answer": "Summarize any conclusions, recommendations, or key takeaways from the document.",
                "question_type": "analysis",
                "difficulty": "medium"
            }
        ]
    
    def score_answer(self, question: str, correct_answer: str, user_answer: str, vectorstore) -> ScoringResult:
        """Score a single answer using LLM"""
        try:
            # Get relevant context for the question
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate scoring prompt
            scoring_input = self.scoring_prompt.format(
                question=question,
                correct_answer=correct_answer,
                user_answer=user_answer,
                context=context
            )
            
            # Get scoring from LLM
            response = self.llm.invoke(scoring_input)
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    scoring_data = json.loads(json_match.group())
                    score = float(scoring_data.get("score", 5.0))
                    feedback = scoring_data.get("feedback", "Score provided")
                else:
                    # Fallback scoring
                    score, feedback = self._fallback_scoring(user_answer, correct_answer)
            except (json.JSONDecodeError, ValueError):
                logger.warning("Failed to parse scoring JSON, using fallback")
                score, feedback = self._fallback_scoring(user_answer, correct_answer)
            
            return ScoringResult(
                question=question,
                user_answer=user_answer,
                correct_answer=correct_answer,
                score=max(0, min(10, score)),  # Ensure score is between 0-10
                feedback=feedback
            )
            
        except Exception as e:
            logger.error(f"Error scoring answer: {e}")
            return ScoringResult(
                question=question,
                user_answer=user_answer,
                correct_answer=correct_answer,
                score=5.0,
                feedback=f"Unable to score automatically. Error: {str(e)}"
            )
    
    def _fallback_scoring(self, user_answer: str, correct_answer: str) -> Tuple[float, str]:
        """Provide fallback scoring when LLM scoring fails"""
        if not user_answer.strip():
            return 0.0, "No answer provided."
        
        # Simple keyword matching for basic scoring
        user_words = set(user_answer.lower().split())
        correct_words = set(correct_answer.lower().split())
        common_words = user_words.intersection(correct_words)
        
        if len(correct_words) > 0:
            similarity = len(common_words) / len(correct_words)
            score = min(10.0, similarity * 10)
        else:
            score = 5.0
        
        return score, f"Basic similarity score based on keyword matching. Consider reviewing the expected answer for a complete response."

class PDFProcessor:
    """Handles PDF processing and vectorization"""
    
    def __init__(self, config: PDFChatbotConfig):
        self.config = config
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=config.GOOGLE_API_KEY
        )
        # Initialize Chroma client with new configuration (no deprecated settings)
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.config.VECTOR_DB_DIR)
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
            
            # Check if collection exists in Chroma
            try:
                collection_name = f"doc_{file_hash}"
                collection = self.chroma_client.get_collection(collection_name)
                is_processed = file_hash in metadata and collection.count() > 0
            except Exception:
                is_processed = False
            
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
            collection_name = f"doc_{file_hash}"
            
            # Try to get existing collection first
            try:
                collection = self.chroma_client.get_collection(collection_name)
                logger.info(f"Collection {collection_name} already exists, deleting and recreating")
                self.chroma_client.delete_collection(collection_name)
            except Exception:
                # Collection doesn't exist, which is fine
                pass
            
            # Create new collection
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Create vectorstore using the new client approach
            vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            
            # Add documents to the vectorstore
            vectorstore.add_documents(documents)
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vectorstore(self, file_hash: str) -> Optional[Chroma]:
        """Load document-specific vector store"""
        try:
            collection_name = f"doc_{file_hash}"
            
            # Check if collection exists
            try:
                collection = self.chroma_client.get_collection(collection_name)
                if collection.count() > 0:
                    # Load existing Chroma instance
                    vectorstore = Chroma(
                        client=self.chroma_client,
                        collection_name=collection_name,
                        embedding_function=self.embeddings
                    )
                    logger.info(f"Vectorstore loaded for document hash: {file_hash}")
                    return vectorstore
                else:
                    logger.warning(f"Collection {collection_name} exists but is empty")
                    return None
            except Exception as e:
                logger.warning(f"Collection {collection_name} does not exist: {e}")
                return None
                
        except Exception as e:
            logger.warning(f"Could not load vectorstore for {file_hash}: {e}")
        return None
    
    def cleanup_old_vectorstores(self, keep_hash: str = None):
        """Clean up old vector stores (optional - for storage management)"""
        try:
            # List all collections
            collections = self.chroma_client.list_collections()
            
            for collection in collections:
                collection_name = collection.name
                if collection_name.startswith("doc_"):
                    # Extract hash from collection name
                    collection_hash = collection_name.replace("doc_", "")
                    if keep_hash is None or collection_hash != keep_hash:
                        try:
                            self.chroma_client.delete_collection(collection_name)
                            logger.info(f"Cleaned up old collection: {collection_name}")
                        except Exception as e:
                            logger.warning(f"Error deleting collection {collection_name}: {e}")
                            
        except Exception as e:
            logger.warning(f"Error cleaning up old vectorstores: {e}")

    def reset_all_collections(self):
        """Reset all collections - useful for complete cleanup"""
        try:
            collections = self.chroma_client.list_collections()
            for collection in collections:
                self.chroma_client.delete_collection(collection.name)
            logger.info("All collections have been reset")
        except Exception as e:
            logger.error(f"Error resetting collections: {e}")

class ChatbotUI:
    """Handles the Streamlit UI"""
    
    def __init__(self):
        self.config = PDFChatbotConfig()
        self.processor = PDFProcessor(self.config)
        self.setup_directories()
        self.initialize_session_state()
    
    def setup_directories(self):
        """Create necessary directories"""
        for directory in [self.config.PDF_DIR, self.config.VECTOR_DB_DIR, self.config.ASSESSMENTS_DIR]:
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
        
        if 'scoring_engine' not in st.session_state:
            st.session_state.scoring_engine = ScoringEngine(self.config, st.session_state.llm)
        
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
        
        # Assessment related state
        if 'current_assessment' not in st.session_state:
            st.session_state.current_assessment = None
        
        if 'assessment_mode' not in st.session_state:
            st.session_state.assessment_mode = False
        
        if 'assessment_questions' not in st.session_state:
            st.session_state.assessment_questions = []
        
        if 'current_question_index' not in st.session_state:
            st.session_state.current_question_index = 0
        
        if 'user_answers' not in st.session_state:
            st.session_state.user_answers = []
        
        if 'assessment_start_time' not in st.session_state:
            st.session_state.assessment_start_time = None
    
    def reset_document_session(self):
        """Reset session for new document"""
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.session_state.vectorstore = None
        st.session_state.qa_chain = None
        st.session_state.current_document = None
        st.session_state.current_document_hash = None
        
        # Reset assessment state
        st.session_state.current_assessment = None
        st.session_state.assessment_mode = False
        st.session_state.assessment_questions = []
        st.session_state.current_question_index = 0
        st.session_state.user_answers = []
        st.session_state.assessment_start_time = None
        
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
    
    def start_assessment(self, difficulty: str, num_questions: int):
        """Start a new assessment"""
        try:
            with st.spinner("🎯 Generating assessment questions..."):
                questions = st.session_state.scoring_engine.generate_questions_for_assessment(
                    st.session_state.vectorstore, 
                    num_questions, 
                    difficulty
                )
                
                if questions:
                    st.session_state.assessment_questions = questions
                    st.session_state.current_question_index = 0
                    st.session_state.user_answers = []
                    st.session_state.assessment_mode = True
                    st.session_state.assessment_start_time = datetime.now()
                    
                    # Generate assessment ID
                    assessment_id = f"assessment_{st.session_state.current_document_hash[:8]}_{int(time.time())}"
                    st.session_state.current_assessment = assessment_id
                    
                    st.success("✅ Assessment generated successfully! You can now start answering questions.")
                    st.rerun()
                else:
                    st.error("Failed to generate assessment questions. Please try again.")
        except Exception as e:
            st.error(f"Error starting assessment: {str(e)}")
            logger.error(f"Assessment generation error: {e}")
    
    def display_assessment_interface(self):
        """Display the assessment interface"""
        if not st.session_state.assessment_mode or not st.session_state.assessment_questions:
            return
        
        current_q_idx = st.session_state.current_question_index
        total_questions = len(st.session_state.assessment_questions)
        current_question = st.session_state.assessment_questions[current_q_idx]
        
        st.subheader(f"📝 Assessment Question {current_q_idx + 1} of {total_questions}")
        
        # Progress bar
        progress = (current_q_idx + 1) / total_questions
        st.progress(progress)
        
        # Display question
        st.markdown(f"**Question:** {current_question['question']}")
        
        # Answer input
        answer_key = f"assessment_answer_{current_q_idx}"
        user_answer = st.text_area(
            "Your Answer:",
            key=answer_key,
            height=150,
            placeholder="Type your answer here..."
        )
        
        # Navigation buttons
        # col1, col2, col3 = st.columns([1, 1, 1])
# Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if current_q_idx > 0:
                if st.button("⬅️ Previous Question"):
                    # Save current answer
                    if user_answer.strip():
                        self.save_current_answer(current_q_idx, user_answer)
                    st.session_state.current_question_index -= 1
                    st.rerun()
        
        with col2:
            if st.button("💾 Save Answer"):
                if user_answer.strip():
                    self.save_current_answer(current_q_idx, user_answer)
                    st.success("Answer saved!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("Please provide an answer before saving.")
        
        with col3:
            if current_q_idx < total_questions - 1:
                if st.button("➡️ Next Question"):
                    if user_answer.strip():
                        self.save_current_answer(current_q_idx, user_answer)
                        st.session_state.current_question_index += 1
                        st.rerun()
                    else:
                        st.warning("Please provide an answer before proceeding.")
            else:
                if st.button("🎯 Finish Assessment", type="primary"):
                    if user_answer.strip():
                        self.save_current_answer(current_q_idx, user_answer)
                        self.complete_assessment()
                    else:
                        st.warning("Please provide an answer for the final question.")
        
        # Show answered questions status
        st.markdown("---")
        st.subheader("📊 Progress Overview")
        answered_count = len([ans for ans in st.session_state.user_answers if ans])
        st.write(f"Questions answered: {answered_count}/{total_questions}")
        
        # Display answered questions grid
        if st.session_state.user_answers:
            cols = st.columns(min(5, total_questions))
            for i in range(total_questions):
                with cols[i % 5]:
                    if i < len(st.session_state.user_answers) and st.session_state.user_answers[i]:
                        st.success(f"Q{i+1} ✅")
                    elif i == current_q_idx:
                        st.info(f"Q{i+1} 📝")
                    else:
                        st.error(f"Q{i+1} ❌")
    
    def save_current_answer(self, question_index: int, answer: str):
        """Save the current answer"""
        # Ensure user_answers list is long enough
        while len(st.session_state.user_answers) <= question_index:
            st.session_state.user_answers.append("")
        
        st.session_state.user_answers[question_index] = answer.strip()
    
    def complete_assessment(self):
        """Complete the assessment and show results"""
        try:
            with st.spinner("🎯 Scoring your assessment..."):
                # Calculate time taken
                time_taken = None
                if st.session_state.assessment_start_time:
                    time_diff = datetime.now() - st.session_state.assessment_start_time
                    time_taken = str(time_diff).split('.')[0]  # Remove microseconds
                
                # Score each question
                scoring_results = []
                total_score = 0
                max_total_score = 0
                
                for i, question_data in enumerate(st.session_state.assessment_questions):
                    user_answer = st.session_state.user_answers[i] if i < len(st.session_state.user_answers) else ""
                    
                    scoring_result = st.session_state.scoring_engine.score_answer(
                        question_data['question'],
                        question_data['correct_answer'],
                        user_answer,
                        st.session_state.vectorstore
                    )
                    
                    scoring_results.append(scoring_result)
                    total_score += scoring_result.score
                    max_total_score += scoring_result.max_score
                
                # Calculate percentage
                percentage = (total_score / max_total_score * 100) if max_total_score > 0 else 0
                
                # Create assessment result
                assessment_result = AssessmentResult(
                    assessment_id=st.session_state.current_assessment,
                    document_name=st.session_state.current_document,
                    total_score=total_score,
                    max_total_score=max_total_score,
                    percentage=percentage,
                    question_results=scoring_results,
                    timestamp=datetime.now().isoformat(),
                    time_taken=time_taken
                )
                
                # Save results
                self.save_assessment_result(assessment_result)
                
                # Display results
                self.display_assessment_results(assessment_result)
                
                # Reset assessment mode
                st.session_state.assessment_mode = False
                
        except Exception as e:
            st.error(f"Error completing assessment: {str(e)}")
            logger.error(f"Assessment completion error: {e}")
    
    def save_assessment_result(self, result: AssessmentResult):
        """Save assessment result to file"""
        try:
            # Load existing results
            existing_results = st.session_state.scoring_engine.load_scoring_results()
            
            # Add new result
            existing_results.append(result)
            
            # Save updated results
            st.session_state.scoring_engine.save_scoring_results(existing_results)
            
            logger.info(f"Assessment result saved: {result.assessment_id}")
        except Exception as e:
            logger.error(f"Error saving assessment result: {e}")
    
    def display_assessment_results(self, result: AssessmentResult):
        """Display assessment results"""
        st.balloons()
        
        st.subheader("🎉 Assessment Complete!")
        
        # Overall score display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Score", f"{result.total_score:.1f}/{result.max_total_score:.1f}")
        with col2:
            st.metric("Percentage", f"{result.percentage:.1f}%")
        with col3:
            if result.time_taken:
                st.metric("Time Taken", result.time_taken)
        
        # Performance indicator
        if result.percentage >= 90:
            st.success("🏆 Excellent Performance!")
        elif result.percentage >= 80:
            st.success("🌟 Good Performance!")
        elif result.percentage >= 70:
            st.info("👍 Satisfactory Performance")
        elif result.percentage >= 60:
            st.warning("📚 Needs Improvement")
        else:
            st.error("🔄 Requires More Study")
        
        # Detailed results
        st.subheader("📋 Detailed Results")
        
        for i, question_result in enumerate(result.question_results):
            with st.expander(f"Question {i+1} - Score: {question_result.score:.1f}/{question_result.max_score:.1f}"):
                st.markdown(f"**Question:** {question_result.question}")
                st.markdown(f"**Your Answer:** {question_result.user_answer}")
                st.markdown(f"**Expected Answer:** {question_result.correct_answer}")
                st.markdown(f"**Score:** {question_result.score:.1f}/{question_result.max_score:.1f}")
                st.markdown(f"**Feedback:** {question_result.feedback}")
        
        # Download results option
        if st.button("📥 Download Results"):
            self.download_assessment_results(result)
    
    def download_assessment_results(self, result: AssessmentResult):
        """Create downloadable assessment results"""
        try:
            # Create a detailed report
            report = f"""
ASSESSMENT RESULTS REPORT
=========================

Document: {result.document_name}
Assessment ID: {result.assessment_id}
Date: {result.timestamp}
Time Taken: {result.time_taken or 'Not recorded'}

OVERALL PERFORMANCE
-------------------
Total Score: {result.total_score:.1f}/{result.max_total_score:.1f}
Percentage: {result.percentage:.1f}%

DETAILED RESULTS
----------------
"""
            
            for i, question_result in enumerate(result.question_results):
                report += f"""
Question {i+1}:
{question_result.question}

Your Answer:
{question_result.user_answer}

Expected Answer:
{question_result.correct_answer}

Score: {question_result.score:.1f}/{question_result.max_score:.1f}
Feedback: {question_result.feedback}

{'='*50}
"""
            
            # Create download button
            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name=f"assessment_report_{result.assessment_id}.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Error creating download: {str(e)}")
    
    def display_assessment_history(self):
        """Display assessment history"""
        try:
            results = st.session_state.scoring_engine.load_scoring_results()
            
            if not results:
                st.info("No assessment history found.")
                return
            
            st.subheader("📊 Assessment History")
            
            # Filter by current document if available
            if st.session_state.current_document:
                doc_results = [r for r in results if r.document_name == st.session_state.current_document]
                if doc_results:
                    st.write(f"Showing results for: **{st.session_state.current_document}**")
                    results = doc_results
                else:
                    st.write("No assessments found for current document.")
                    return
            
            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x.timestamp, reverse=True)
            
            for result in results:
                with st.expander(f"📅 {result.timestamp[:19]} - Score: {result.percentage:.1f}%"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Score", f"{result.total_score:.1f}/{result.max_total_score:.1f}")
                    with col2:
                        st.metric("Percentage", f"{result.percentage:.1f}%")
                    with col3:
                        st.metric("Questions", len(result.question_results))
                    
                    if st.button(f"View Details", key=f"view_{result.assessment_id}"):
                        self.display_assessment_results(result)
        
        except Exception as e:
            st.error(f"Error loading assessment history: {str(e)}")
    
    def run(self):
        """Main application runner"""
        # Page configuration
        st.set_page_config(
            page_title="📚 PDF Assistant with Automated Scoring",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("📚 PDF Assistant with Automated Scoring")
        st.markdown("Upload a PDF document to chat with it and take automated assessments!")
        
        # Sidebar
        with st.sidebar:
            st.header("📁 Document Management")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload PDF Document",
                type=self.config.SUPPORTED_FILE_TYPES,
                help=f"Maximum file size: {self.config.MAX_FILE_SIZE_MB}MB"
            )
            
            # Process uploaded file
            if uploaded_file is not None:
                if self.validate_uploaded_file(uploaded_file):
                    # Check if this is a new file or the same file
                    if (st.session_state.current_document != uploaded_file.name or 
                        st.session_state.vectorstore is None):
                        with st.spinner("Processing document..."):
                            self.process_uploaded_file(uploaded_file)
                    
                    st.success(f"✅ Document loaded: {uploaded_file.name}")
                    
                    # Document info
                    if st.session_state.current_document_hash:
                        metadata = self.processor.load_metadata()
                        if st.session_state.current_document_hash in metadata:
                            doc_info = metadata[st.session_state.current_document_hash]
                            st.info(f"Chunks: {doc_info.get('chunks_count', 'Unknown')}")
            
            # Quick Actions
            if st.session_state.vectorstore is not None:
                st.markdown("---")
                st.header("🚀 Quick Actions")
                
                if st.button("📝 Summarize Document"):
                    prompt = self.generate_automated_response("summarize")
                    st.session_state.chat_history.append(("user", prompt))
                    with st.spinner("Generating summary..."):
                        try:
                            response = st.session_state.qa_chain.run(prompt)
                            st.session_state.chat_history.append(("assistant", response))
                        except Exception as e:
                            st.error(f"Error generating summary: {str(e)}")
                    st.rerun()
                
                if st.button("📚 Create Study Guide"):
                    prompt = self.generate_automated_response("study_guide")
                    st.session_state.chat_history.append(("user", prompt))
                    with st.spinner("Creating study guide..."):
                        try:
                            response = st.session_state.qa_chain.run(prompt)
                            st.session_state.chat_history.append(("assistant", response))
                        except Exception as e:
                            st.error(f"Error creating study guide: {str(e)}")
                    st.rerun()
                
                if st.button("❓ Generate Sample Questions"):
                    prompt = self.generate_automated_response("exam_questions")
                    st.session_state.chat_history.append(("user", prompt))
                    with st.spinner("Generating questions..."):
                        try:
                            response = st.session_state.qa_chain.run(prompt)
                            st.session_state.chat_history.append(("assistant", response))
                        except Exception as e:
                            st.error(f"Error generating questions: {str(e)}")
                    st.rerun()
                
                # Assessment Section
                st.markdown("---")
                st.header("🎯 Automated Assessment")
                
                if not st.session_state.assessment_mode:
                    difficulty = st.selectbox(
                        "Select Difficulty:",
                        ["easy", "medium", "hard"],
                        index=1
                    )
                    
                    num_questions = st.slider(
                        "Number of Questions:",
                        min_value=3,
                        max_value=10,
                        value=5
                    )
                    
                    if st.button("🎯 Start Assessment", type="primary"):
                        self.start_assessment(difficulty, num_questions)
                else:
                    st.info("📝 Assessment in progress...")
                    if st.button("❌ Cancel Assessment"):
                        st.session_state.assessment_mode = False
                        st.session_state.assessment_questions = []
                        st.session_state.user_answers = []
                        st.rerun()
                
                # Assessment History
                if st.button("📊 View Assessment History"):
                    st.session_state.show_history = True
        
        # Main content area
        if st.session_state.vectorstore is None:
            # Welcome message
            st.markdown("""
            ## Welcome to PDF Assistant with Automated Scoring! 🎉
            
            This application allows you to:
            - 📄 Upload and chat with PDF documents
            - 🤖 Get AI-powered answers to your questions
            - 🎯 Take automated assessments based on the document content
            - 📊 Track your learning progress with detailed scoring
            
            **To get started:**
            1. Upload a PDF document using the sidebar
            2. Wait for the document to be processed
            3. Start chatting or take an assessment!
            """)
        else:
            # Check if we should show assessment interface
            if st.session_state.assessment_mode:
                self.display_assessment_interface()
            elif getattr(st.session_state, 'show_history', False):
                self.display_assessment_history()
                if st.button("🔙 Back to Chat"):
                    st.session_state.show_history = False
                    st.rerun()
            else:
                # Regular chat interface
                # Display chat history
                for role, message in st.session_state.chat_history:
                    if role == "user":
                        with st.chat_message("user"):
                            st.write(message)
                    else:
                        with st.chat_message("assistant"):
                            st.write(message)
                
                # Chat input
                if user_question := st.chat_input("Ask a question about the document..."):
                    # Add user message to chat history
                    st.session_state.chat_history.append(("user", user_question))
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.write(user_question)
                    
                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        
                        try:
                            # Create callback handler for streaming
                            callback_handler = StreamlitCallbackHandler(message_placeholder)
                            
                            # Get response
                            with st.spinner("Thinking..."):
                                response = st.session_state.qa_chain.run(
                                    user_question,
                                    callbacks=[callback_handler]
                                )
                            
                            # Final response
                            message_placeholder.markdown(response)
                            
                            # Add assistant response to chat history
                            st.session_state.chat_history.append(("assistant", response))
                            
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            message_placeholder.error(error_msg)
                            st.session_state.chat_history.append(("assistant", error_msg))
                            logger.error(f"Error in chat: {e}")
        
        # Footer
        st.markdown("---")
        st.markdown("🤖 Powered by Google Gemini AI | Built with ❤️ using Streamlit")

def main():
    """Main entry point"""
    try:
        app = ChatbotUI()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application startup error: {e}")

if __name__ == "__main__":
    main()