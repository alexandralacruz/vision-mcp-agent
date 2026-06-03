# agent/agent.py
"""
Agente de visión basado en Ollama con soporte MCP.
El agente recibe una consulta en lenguaje natural, decide qué herramientas
MCP usar, las invoca y sintetiza la respuesta final.

Flujo:
  1. Usuario describe lo que recuerda de una imagen.
  2. El agente llama clip_search para obtener candidatos.
  3. Si hay ambigüedad, llama vqa_query para refinar.
  4. Retorna los resultados con explicación.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx
from config.settings import settings

logger = logging.getLogger(__name__)

# ── System prompt del agente ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a visual memory search agent. Your job is to help users find images in a repository based on what they remember seeing.

You have access to the following tools:
- clip_search: Search images by text description using CLIP embeddings
- vqa_query: Ask a question about a specific image using BLIP VQA
- list_repository: List all indexed images
- get_image_info: Get detailed information about a specific image
- search_by_image: Find visually similar images using a reference image

STRATEGY:
1. Start with clip_search using the user's description
2. If results are uncertain, use vqa_query to verify specific details
3. Combine visual similarity scores with VQA answers to rank results
4. Always explain WHY you think an image matches the description

Respond in the same language as the user. Be concise and helpful.
When presenting results, always include the image_id and similarity score.
"""

# ── Definiciones de herramientas para Ollama ─────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "clip_search",
            "description": "Search images in repository by text description using CLIP embeddings",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language description of the image"},
                    "top_k": {"type": "integer", "description": "Max results to return", "default": 5},
                    "threshold": {"type": "number", "description": "Min similarity threshold 0-1", "default": 0.20},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vqa_query",
            "description": "Ask a visual question about a specific image using BLIP VQA",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string", "description": "Image ID in repository"},
                    "question": {"type": "string", "description": "Question about the image content"},
                },
                "required": ["image_id", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_repository",
            "description": "List indexed images in the repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                    "offset": {"type": "integer", "default": 0},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_image_info",
            "description": "Get detailed metadata of a specific image",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                },
                "required": ["image_id"],
            },
        },
    },
]


# ── Cliente Ollama ────────────────────────────────────────────────────────────

class OllamaClient:
    """Cliente HTTP para la API de Ollama."""

    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> dict:
        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": settings.OLLAMA_TEMPERATURE,
                "num_predict": settings.OLLAMA_MAX_TOKENS,
            },
        }
        if tools:
            payload["tools"] = tools

        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def is_available(self) -> bool:
        try:
            r = await self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self.client.aclose()


# ── Agente con loop MCP ───────────────────────────────────────────────────────

class VisionAgent:
    """
    Agente que orquesta las herramientas MCP para búsqueda de imágenes.
    Implementa el loop: LLM → tool_call → tool_result → LLM.
    """

    def __init__(self):
        self.ollama = OllamaClient()
        self._mcp_tool_handlers = self._build_handlers()

    def _build_handlers(self) -> dict:
        """Mapea nombre de herramienta → función async de ejecución."""
        from repository.manager import get_repository
        from models.blip_model import get_blip_model

        repo = get_repository()
        blip = get_blip_model()

        async def _clip_search(args: dict) -> str:
            results = repo.search_by_text(
                args["query"],
                top_k=args.get("top_k", 5),
                threshold=args.get("threshold", 0.20),
            )
            if not results:
                return json.dumps({"status": "no_results", "results": []})
            simplified = [
                {
                    "image_id": r["image_id"],
                    "score": r["score"],
                    "filename": r["metadata"]["filename"],
                    "description": r["metadata"]["searchable_text"][:400],
                    "tags": r["metadata"]["tags"],
                }
                for r in results
            ]
            return json.dumps({"status": "ok", "total": len(results), "results": simplified}, ensure_ascii=False)

        async def _vqa_query(args: dict) -> str:
            from config.settings import settings
            meta = repo.get_image(args["image_id"])
            if not meta:
                return json.dumps({"status": "error", "message": "Image not found"})
            img_path = settings.REPOSITORY_PATH / meta.filename
            answer = blip.answer_custom_question(img_path, args["question"])
            return json.dumps({"status": "ok", "answer": answer})

        async def _list_repository(args: dict) -> str:
            images = repo.list_images(limit=args.get("limit", 10), offset=args.get("offset", 0))
            summary = [
                {"image_id": img["image_id"], "filename": img["filename"], "tags": img["tags"]}
                for img in images
            ]
            return json.dumps({"total": repo.total_images, "images": summary}, ensure_ascii=False)

        async def _get_image_info(args: dict) -> str:
            meta = repo.get_image(args["image_id"])
            if not meta:
                return json.dumps({"status": "error", "message": "Not found"})
            return json.dumps({"status": "ok", "metadata": meta.to_dict()}, ensure_ascii=False)

        return {
            "clip_search": _clip_search,
            "vqa_query": _vqa_query,
            "list_repository": _list_repository,
            "get_image_info": _get_image_info,
        }

    async def run(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
        max_iterations: int = 5,
    ) -> dict:
        """
        Ejecuta el agente con el mensaje del usuario.

        Args:
            user_message: Consulta del usuario.
            conversation_history: Historial previo de conversación.
            max_iterations: Máximo de ciclos herramienta → LLM.

        Returns:
            {
                "response": str,           # Respuesta final del agente
                "tool_calls": list,        # Herramientas invocadas
                "search_results": list,    # Resultados de búsqueda
                "iterations": int,
            }
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_message})

        tool_calls_log = []
        search_results = []
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            logger.info(f"[Agent] Iteración {iterations}")

            response = await self.ollama.chat(messages=messages, tools=TOOLS)
            msg = response.get("message", {})

            # Si el modelo quiere llamar herramientas
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls})

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    tool_args = fn.get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except Exception:
                            tool_args = {}

                    logger.info(f"[Agent] Llamando herramienta: {tool_name}({tool_args})")
                    tool_calls_log.append({"tool": tool_name, "args": tool_args})

                    handler = self._mcp_tool_handlers.get(tool_name)
                    if handler:
                        try:
                            result_str = await handler(tool_args)
                            result_data = json.loads(result_str)

                            # Capturar resultados de búsqueda
                            if tool_name == "clip_search" and result_data.get("results"):
                                search_results.extend(result_data["results"])
                        except Exception as e:
                            result_str = json.dumps({"error": str(e)})
                            logger.error(f"[Agent] Error en herramienta {tool_name}: {e}")
                    else:
                        result_str = json.dumps({"error": f"Tool '{tool_name}' not implemented"})

                    messages.append({"role": "tool", "content": result_str})

            else:
                # El modelo tiene la respuesta final
                final_response = msg.get("content", "")
                logger.info(f"[Agent] Respuesta final generada tras {iterations} iteraciones.")
                return {
                    "response": final_response,
                    "tool_calls": tool_calls_log,
                    "search_results": search_results,
                    "iterations": iterations,
                }

        # Agotamos iteraciones, pedir resumen
        messages.append({
            "role": "user",
            "content": "Por favor resume los resultados encontrados hasta ahora.",
        })
        response = await self.ollama.chat(messages=messages)
        return {
            "response": response.get("message", {}).get("content", "No se obtuvo respuesta."),
            "tool_calls": tool_calls_log,
            "search_results": search_results,
            "iterations": iterations,
        }

    async def close(self):
        await self.ollama.close()


# Singleton del agente
_agent: VisionAgent | None = None


def get_agent() -> VisionAgent:
    global _agent
    if _agent is None:
        _agent = VisionAgent()
    return _agent
