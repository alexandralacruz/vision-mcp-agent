# api/main.py
"""
API FastAPI del Vision MCP Agent.
Endpoints:
  POST /search          → Búsqueda con agente Ollama + MCP
  POST /search/direct   → Búsqueda directa CLIP sin agente
  POST /vqa             → Visual Question Answering
  POST /repository/add  → Agregar imagen al repositorio
  GET  /repository      → Listar imágenes
  GET  /repository/{id} → Detalle de imagen
  DELETE /repository/{id} → Eliminar imagen
  GET  /images/{id}     → Servir imagen
  GET  /health          → Estado del sistema
"""
from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from config.settings import settings
from api.schemas import (
    SearchRequest,
    VQARequest,
    AddImageRequest,
    AgentResponse,
    RepositoryStats,
)
from agent.agent import get_agent
from repository.manager import get_repository
from models.blip_model import get_blip_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicialización y cleanup de recursos."""
    logger.info("Iniciando Vision MCP Agent API...")
    # Pre-cargar modelos en background
    try:
        _ = get_repository()
        logger.info("Repositorio listo.")
    except Exception as e:
        logger.error(f"Error inicializando repositorio: {e}")
    yield
    # Cleanup
    agent = get_agent()
    await agent.close()
    logger.info("API detenida.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Agente de búsqueda visual con CLIP, BLIP VQA y Ollama via MCP",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Búsqueda ─────────────────────────────────────────────────────────────────

@app.post("/search", response_model=AgentResponse, tags=["Búsqueda"])
async def search_with_agent(request: SearchRequest):
    """
    Búsqueda inteligente usando el agente Ollama + herramientas MCP.
    El agente decide qué herramientas usar y sintetiza la respuesta.
    """
    if request.use_agent:
        logger.info(f"[Search] Usando AGENTE Ollama + MCP para query: {request.query[:80]}")
        agent = get_agent()
        # Verificar Ollama disponible
        if not await agent.ollama.is_available():
            raise HTTPException(
                status_code=503,
                detail="Ollama no está disponible. Verifique que esté corriendo en localhost:11434",
            )
        result = await agent.run(
            user_message=request.query,
            conversation_history=request.conversation_history or None,
        )
        return AgentResponse(**result)
    else:
        # Búsqueda directa sin agente
        logger.info(f"[Search] Usando CLIP DIRECTO para query: {request.query[:80]}")
        repo = get_repository()
        results = repo.search_by_text(
            request.query,
            top_k=request.top_k,
            threshold=request.threshold,
        )
        return AgentResponse(
            response=f"Se encontraron {len(results)} imágenes para: '{request.query}'",
            tool_calls=[{"tool": "clip_search", "args": {"query": request.query}}],
            search_results=[
                {
                    "image_id": r["image_id"],
                    "score": r["score"],
                    "filename": r["metadata"]["filename"],
                    "description": r["metadata"]["searchable_text"][:200],
                    "tags": r["metadata"]["tags"],
                }
                for r in results
            ],
            iterations=1,
        )


@app.post("/search/upload", tags=["Búsqueda"])
async def search_by_uploaded_image(
    file: UploadFile = File(...),
    top_k: int = Form(5),
):
    """Busca imágenes similares a una imagen subida (búsqueda visual directa)."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    # Guardar temporalmente
    tmp_path = Path(f"/tmp/query_{file.filename}")
    content = await file.read()
    tmp_path.write_bytes(content)

    repo = get_repository()
    results = repo.search_by_image(tmp_path, top_k=top_k)

    return {
        "total": len(results),
        "results": [
            {
                "image_id": r["image_id"],
                "score": r["score"],
                "filename": r["metadata"]["filename"],
                "tags": r["metadata"]["tags"],
            }
            for r in results
        ],
    }


# ── VQA ──────────────────────────────────────────────────────────────────────

@app.post("/vqa", tags=["VQA"])
async def visual_question_answering(request: VQARequest):
    """
    Responde una pregunta sobre una imagen específica usando BLIP VQA.
    """
    repo = get_repository()
    meta = repo.get_image(request.image_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Imagen no encontrada: {request.image_id}")

    img_path = settings.REPOSITORY_PATH / meta.filename
    blip = get_blip_model()
    answer = blip.answer_custom_question(img_path, request.question)

    return {
        "image_id": request.image_id,
        "question": request.question,
        "answer": answer,
        "filename": meta.filename,
    }


# ── Repositorio ───────────────────────────────────────────────────────────────

@app.post("/repository/add", tags=["Repositorio"])
async def add_image_to_repository(
    file: UploadFile = File(...),
    tags: str = Form(""),
):
    """
    Agrega y indexa una imagen al repositorio.
    Genera automáticamente embeddings CLIP y descripción VQA.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado. Use: {settings.SUPPORTED_FORMATS}",
        )

    # Guardar temporalmente
    tmp_path = Path(f"/tmp/upload_{file.filename}")
    content = await file.read()
    tmp_path.write_bytes(content)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    repo = get_repository()
    try:
        meta = repo.add_image(tmp_path, tags=tag_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexando imagen: {str(e)}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "status": "indexed",
        "image_id": meta.image_id,
        "filename": meta.filename,
        "dimensions": f"{meta.width}x{meta.height}",
        "tags": meta.tags,
        "searchable_text_preview": meta.searchable_text[:200],
    }


@app.get("/repository", tags=["Repositorio"])
async def list_repository(limit: int = 20, offset: int = 0):
    """Lista las imágenes del repositorio con paginación."""
    repo = get_repository()
    images = repo.list_images(limit=limit, offset=offset)
    return {
        "total": repo.total_images,
        "limit": limit,
        "offset": offset,
        "images": images,
    }


@app.get("/repository/{image_id}", tags=["Repositorio"])
async def get_image_info(image_id: str):
    """Obtiene metadatos completos de una imagen."""
    repo = get_repository()
    meta = repo.get_image(image_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Imagen no encontrada: {image_id}")
    return meta.to_dict()


@app.delete("/repository/{image_id}", tags=["Repositorio"])
async def delete_image(image_id: str):
    """Elimina una imagen del repositorio."""
    repo = get_repository()
    deleted = repo.delete_image(image_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Imagen no encontrada: {image_id}")
    return {"status": "deleted", "image_id": image_id}


# ── Servir imágenes ───────────────────────────────────────────────────────────

@app.get("/images/{image_id}", tags=["Imágenes"])
async def serve_image(image_id: str):
    """Sirve la imagen original del repositorio."""
    repo = get_repository()
    meta = repo.get_image(image_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    img_path = settings.REPOSITORY_PATH / meta.filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Archivo de imagen no encontrado")

    content = img_path.read_bytes()
    suffix = img_path.suffix.lower().lstrip(".")
    media_type = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"

    return StreamingResponse(io.BytesIO(content), media_type=media_type)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
async def health_check():
    """Estado del sistema."""
    repo = get_repository()
    agent = get_agent()
    ollama_ok = await agent.ollama.is_available()

    return {
        "status": "ok",
        "ollama_available": ollama_ok,
        "ollama_model": settings.OLLAMA_MODEL,
        "repository_images": repo.total_images,
        "clip_model": settings.CLIP_MODEL_NAME,
        "blip_model": settings.BLIP_MODEL_NAME,
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info",
    )
