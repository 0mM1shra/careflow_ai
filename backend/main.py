import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database.db import init_db
from backend.api.routes import router
from backend.config import settings

from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("careflow")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing SQLite database tables...")
    init_db()
    logger.info(f"CareFlow API ready on {settings.BACKEND_HOST}:{settings.BACKEND_PORT}")
    yield

# Initialize FastAPI App
app = FastAPI(
    title="CareFlow API",
    description="MedGemma-Powered Clinical Intake & Reviewer Backend",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit frontend and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Router
app.include_router(router)

@app.get("/")
def root():
    return {
        "app": "CareFlow Clinical Intake and Reviewer System",
        "status": "online",
        "documentation": "/docs",
        "model_mode": settings.MEDGEMMA_MODE,
        "triage_protocol_version": settings.TRIAGE_RULE_VERSION
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True,
        reload_dirs=["backend"],
        reload_excludes=[".venv", ".git", "__pycache__", "*.pyc", "*.log"]
    )
