# config/settings.py
from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    # ── Ollama ──────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"          # Modelo con visión
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_MAX_TOKENS: int = 2048

    # ── MCP Server ───────────────────────────────────────────────────────────
    MCP_SERVER_HOST: str = "localhost"
    MCP_SERVER_PORT: int = 8001
    MCP_SERVER_NAME: str = "vision-search-mcp"
    MCP_SERVER_VERSION: str = "1.0.0"

    # ── FastAPI ──────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "Vision MCP Agent API"
    API_VERSION: str = "1.0.0"

    # ── CLIP ─────────────────────────────────────────────────────────────────
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    CLIP_DEVICE: str = "cpu"                       # "cuda" si tienes GPU
    CLIP_EMBEDDING_DIM: int = 512
    CLIP_SIMILARITY_THRESHOLD: float = 0.20

    # ── BLIP VQA ─────────────────────────────────────────────────────────────
    BLIP_MODEL_NAME: str = "Salesforce/blip-vqa-base"
    BLIP_DEVICE: str = "cpu"
    BLIP_MAX_ANSWER_LENGTH: int = 50

    # ── Repositorio ──────────────────────────────────────────────────────────
    REPOSITORY_PATH: Path = Path("repository/storage/images")
    EMBEDDINGS_PATH: Path = Path("repository/storage/embeddings")
    INDEX_PATH: Path = Path("repository/storage/index.json")
    MAX_RESULTS: int = 10
    SUPPORTED_FORMATS: list = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    # ── Streamlit ────────────────────────────────────────────────────────────
    STREAMLIT_API_URL: str = "http://api:8001"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Crear directorios si no existen
settings.REPOSITORY_PATH.mkdir(parents=True, exist_ok=True)
settings.EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)
