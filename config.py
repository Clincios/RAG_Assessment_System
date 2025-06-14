import os
from pathlib import Path

class SharedConfig:
    """Shared configuration for all RAG applications"""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAkHRbkUvKvnfzWpoX1pks8hNUc78PXqXs")
    
    # Directory paths
    BASE_DIR = Path(__file__).parent
    PDF_DIR = BASE_DIR / "pdfFiles"
    VECTOR_DB_DIR = BASE_DIR / "vectorDB"
    METADATA_FILE = BASE_DIR / "pdf_metadata.json"
    
    # Document processing settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_MEMORY_MESSAGES = 10
    SUPPORTED_FILE_TYPES = ["pdf"]
    MAX_FILE_SIZE_MB = 50
    
    # Model settings
    EMBEDDING_MODEL = "models/embedding-001"
    LLM_MODEL = "gemini-1.5-flash"
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 4096
    
    # Prompt template
    DEFAULT_PROMPT_TEMPLATE = """You are an intelligent document assistant specialized in analyzing and answering questions about PDF documents. 
    You provide accurate, detailed, and helpful responses based on the provided context.

    Instructions:
    - Answer questions directly and accurately based on the context
    - If information is not available in the context, clearly state that
    - Provide specific page references when possible
    - Use professional and informative tone
    - For complex queries, break down your response into clear sections

    Context: {context}
    Chat History: {history}

    Question: {question}
    Assistant:"""
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.PDF_DIR, cls.VECTOR_DB_DIR]:
            directory.mkdir(parents=True, exist_ok=True) 