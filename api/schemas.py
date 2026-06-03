# api/schemas.py
"""Modelos Pydantic para la API."""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class SearchRequest(BaseModel):
    query: str = Field(..., description="Descripción de la imagen buscada")
    top_k: int = Field(5, ge=1, le=50, description="Número de resultados")
    threshold: float = Field(0.20, ge=0.0, le=1.0, description="Umbral de similitud")
    use_agent: bool = Field(True, description="Usar agente Ollama (True) o búsqueda directa (False)")
    conversation_history: list[dict] = Field(default_factory=list)


class VQARequest(BaseModel):
    image_id: str
    question: str


class AddImageRequest(BaseModel):
    image_path: str = Field(..., description="Ruta local a la imagen")
    tags: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    image_id: str
    score: float
    filename: str
    description: Optional[str] = None
    tags: list[str] = []
    image_url: Optional[str] = None


class AgentResponse(BaseModel):
    response: str
    tool_calls: list[dict]
    search_results: list[dict]
    iterations: int


class RepositoryStats(BaseModel):
    total_images: int
    storage_path: str
