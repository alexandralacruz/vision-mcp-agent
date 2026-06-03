# mcp_server/server.py
"""
Servidor MCP (Model Context Protocol) para el agente de visión.
Expone herramientas que el agente Ollama puede invocar:
  - clip_search      → búsqueda semántica por texto
  - vqa_query        → Visual Question Answering sobre una imagen del repo
  - image_search_by_image → búsqueda por imagen similar
  - list_repository  → listar imágenes indexadas
  - get_image_info   → detalle de una imagen específica
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    CallToolResult,
    ListToolsResult,
)

from config.settings import settings
from repository.manager import get_repository
from models.clip_model import get_clip_model
from models.blip_model import get_blip_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Instancia del servidor MCP ───────────────────────────────────────────────
app = Server(settings.MCP_SERVER_NAME)


# ── Definición de herramientas ───────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    """Registra todas las herramientas disponibles en el servidor MCP."""
    return [
        Tool(
            name="clip_search",
            description=(
                "Busca imágenes en el repositorio usando una descripción textual. "
                "Usa embeddings CLIP para encontrar imágenes semánticamente similares "
                "a la descripción proporcionada. Ideal cuando el usuario recuerda "
                "características visuales de una imagen."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Descripción en lenguaje natural de la imagen buscada",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Número máximo de resultados (default: 5)",
                        "default": 5,
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Umbral mínimo de similitud 0-1 (default: 0.20)",
                        "default": 0.20,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="vqa_query",
            description=(
                "Hace una pregunta en lenguaje natural sobre una imagen específica "
                "del repositorio usando BLIP VQA. Útil para verificar detalles "
                "o extraer información específica de una imagen encontrada."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_id": {
                        "type": "string",
                        "description": "ID de la imagen en el repositorio",
                    },
                    "question": {
                        "type": "string",
                        "description": "Pregunta sobre el contenido de la imagen (en inglés para mejor precisión)",
                    },
                },
                "required": ["image_id", "question"],
            },
        ),
        Tool(
            name="search_by_image",
            description=(
                "Busca imágenes visualmente similares a una imagen de referencia "
                "usando embeddings CLIP. La imagen de referencia se provee en base64."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "Imagen de consulta codificada en base64",
                    },
                    "image_format": {
                        "type": "string",
                        "description": "Formato de la imagen (jpeg, png, etc.)",
                        "default": "jpeg",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Número máximo de resultados",
                        "default": 5,
                    },
                },
                "required": ["image_base64"],
            },
        ),
        Tool(
            name="list_repository",
            description="Lista las imágenes indexadas en el repositorio con sus metadatos.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Máximo de imágenes a retornar",
                        "default": 20,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Desplazamiento para paginación",
                        "default": 0,
                    },
                },
            },
        ),
        Tool(
            name="get_image_info",
            description="Obtiene información detallada de una imagen específica del repositorio, incluyendo descripción VQA completa.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_id": {
                        "type": "string",
                        "description": "ID de la imagen",
                    },
                },
                "required": ["image_id"],
            },
        ),
    ]


#  Implementación de herramientas 

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent]:
    """Dispatcher de herramientas MCP."""

    repo = get_repository()

    #  clip_search 
    if name == "clip_search":
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)
        threshold = arguments.get("threshold", settings.CLIP_SIMILARITY_THRESHOLD)

        logger.info(f"[MCP] clip_search: '{query}' top_k={top_k}")
        results = repo.search_by_text(query, top_k=top_k, threshold=threshold)

        if not results:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "no_results",
                    "message": f"No se encontraron imágenes que coincidan con: '{query}'",
                    "query": query,
                    "results": [],
                }, ensure_ascii=False),
            )]

        # Simplificar resultados para el agente
        simplified = [
            {
                "image_id": r["image_id"],
                "score": r["score"],
                "filename": r["metadata"]["filename"],
                "searchable_text": r["metadata"]["searchable_text"][:300],
                "tags": r["metadata"]["tags"],
                "indexed_at": r["metadata"]["indexed_at"],
            }
            for r in results
        ]

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "ok",
                "query": query,
                "total_found": len(results),
                "results": simplified,
            }, ensure_ascii=False, indent=2),
        )]

    #  vqa_query 
    elif name == "vqa_query":
        image_id = arguments["image_id"]
        question = arguments["question"]

        meta = repo.get_image(image_id)
        if not meta:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": f"Imagen no encontrada: {image_id}"}),
            )]

        image_path = settings.REPOSITORY_PATH / meta.filename
        blip = get_blip_model()

        logger.info(f"[MCP] vqa_query: {image_id} - '{question}'")
        answer = blip.answer_custom_question(image_path, question)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "ok",
                "image_id": image_id,
                "question": question,
                "answer": answer,
            }, ensure_ascii=False),
        )]

    #  search_by_image 
    elif name == "search_by_image":
        import io
        from PIL import Image

        image_b64 = arguments["image_base64"]
        fmt = arguments.get("image_format", "jpeg")
        top_k = arguments.get("top_k", 5)

        # Decodificar imagen
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Guardar temporalmente
        tmp_path = Path("/tmp/mcp_query_image.jpg")
        img.save(str(tmp_path), format="JPEG")

        logger.info(f"[MCP] search_by_image top_k={top_k}")
        results = repo.search_by_image(tmp_path, top_k=top_k)

        simplified = [
            {
                "image_id": r["image_id"],
                "score": r["score"],
                "filename": r["metadata"]["filename"],
                "searchable_text": r["metadata"]["searchable_text"][:300],
            }
            for r in results
        ]

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "ok",
                "total_found": len(results),
                "results": simplified,
            }, ensure_ascii=False, indent=2),
        )]

    #  list_repository 
    elif name == "list_repository":
        limit = arguments.get("limit", 20)
        offset = arguments.get("offset", 0)

        images = repo.list_images(limit=limit, offset=offset)
        summary = [
            {
                "image_id": img["image_id"],
                "filename": img["filename"],
                "tags": img["tags"],
                "indexed_at": img["indexed_at"],
                "dimensions": f"{img['width']}x{img['height']}",
            }
            for img in images
        ]

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "ok",
                "total_in_repository": repo.total_images,
                "returned": len(summary),
                "images": summary,
            }, ensure_ascii=False, indent=2),
        )]

    #  get_image_info 
    elif name == "get_image_info":
        image_id = arguments["image_id"]
        meta = repo.get_image(image_id)

        if not meta:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": f"Imagen no encontrada: {image_id}"}),
            )]

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "ok",
                "metadata": meta.to_dict(),
            }, ensure_ascii=False, indent=2),
        )]

    else:
        return [TextContent(
            type="text",
            text=json.dumps({"status": "error", "message": f"Herramienta desconocida: {name}"}),
        )]


#  Punto de entrada 

async def main():
    logger.info(f"Iniciando servidor MCP '{settings.MCP_SERVER_NAME}'...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=settings.MCP_SERVER_NAME,
                server_version=settings.MCP_SERVER_VERSION,
                capabilities=app.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
