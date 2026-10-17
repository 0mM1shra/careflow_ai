import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory for CareFlow
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

class Settings:
    # Model execution mode: 'mock', 'hosted', or 'local'
    MEDGEMMA_MODE: str = os.getenv("MEDGEMMA_MODE", "mock").lower()
    
    # Model Identifier
    MEDGEMMA_MODEL_ID: str = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it")
    
    # Hosted Endpoint & API Key
    MEDGEMMA_ENDPOINT: str = os.getenv("MEDGEMMA_ENDPOINT", "")
    MEDGEMMA_API_KEY: str = os.getenv("MEDGEMMA_API_KEY", "")
    
    # Local Quantization
    MEDGEMMA_LOCAL_QUANTIZATION: str = os.getenv("MEDGEMMA_LOCAL_QUANTIZATION", "4bit")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'careflow.db'}")
    
    # Workflow Limits
    MAX_FOLLOWUP_QUESTIONS: int = int(os.getenv("MAX_FOLLOWUP_QUESTIONS", "8"))
    
    # Server network
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "127.0.0.1")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
    
    # Triage protocol version
    TRIAGE_RULE_VERSION: str = "v1.0.0-prototype"

settings = Settings()
