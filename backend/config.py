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
    
    # Hosted Endpoint & API Key (supports MEDGEMMA_API_KEY or standard HF_TOKEN)
    MEDGEMMA_ENDPOINT: str = os.getenv("MEDGEMMA_ENDPOINT", "")
    MEDGEMMA_API_KEY: str = os.getenv("MEDGEMMA_API_KEY", os.getenv("HF_TOKEN", ""))
    
    # Local Quantization: '4bit', '8bit', or 'none'
    MEDGEMMA_LOCAL_QUANTIZATION: str = os.getenv("MEDGEMMA_LOCAL_QUANTIZATION", "4bit")
    
    # Model timeouts and retries
    MODEL_TIMEOUT_SECONDS: int = int(os.getenv("MODEL_TIMEOUT_SECONDS", "45"))
    MODEL_MAX_RETRIES: int = int(os.getenv("MODEL_MAX_RETRIES", "1"))

    # ASR / Voice Transcription Configuration
    ASR_MODE: str = os.getenv("ASR_MODE", "auto").lower()              # 'auto', 'local', 'mock'
    ASR_MODEL_SIZE: str = os.getenv("ASR_MODEL_SIZE", "base").lower()  # 'base', 'tiny', 'small'
    ASR_DEVICE: str = os.getenv("ASR_DEVICE", "cpu").lower()            # 'cpu' or 'cuda'

    # OCR Engine Configuration: 'auto', 'winocr', 'paddleocr'
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "auto").lower()

    # Cache and Storage Directories (e.g. D:\huggingface on Windows)
    HF_HOME: str = os.getenv("HF_HOME", "")
    CAREFLOW_TEMP_DIR: str = os.getenv("CAREFLOW_TEMP_DIR", "")

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

    @classmethod
    def get_masked_key(cls, key: str) -> str:
        """Returns safe masked representation of an API key for logs/diagnostics."""
        if not key:
            return "none"
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"

# Set HF cache environment variable if configured
if os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = os.getenv("HF_HOME")

settings = Settings()
