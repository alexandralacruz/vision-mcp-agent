# models/blip_model.py
"""
Wrapper para BLIP (Bootstrapping Language-Image Pre-training).
Implementa Visual Question Answering (VQA): dada una imagen y una pregunta
en lenguaje natural, devuelve una respuesta contextual.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union
from functools import lru_cache

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

from config.settings import settings

logger = logging.getLogger(__name__)

# Preguntas VQA estándar para enriquecer el índice de imágenes
INDEXING_QUESTIONS = [
    "What is the main subject of this image?",
    "What colors are predominant in this image?",
    "What is the setting or environment shown?",
    "Are there people in this image? If so, what are they doing?",
    "What objects are visible in the foreground?",
    "What time of day does this image appear to be taken?",
    "What is the overall mood or atmosphere of this image?",
    "Is this image indoors or outdoors?",
    "What action or activity is taking place?",
    "What style or category does this image belong to?",
]


class BLIPWrapper:
    """Singleton wrapper para el modelo BLIP-VQA."""

    _instance: "BLIPWrapper | None" = None

    def __new__(cls) -> "BLIPWrapper":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        logger.info(f"Cargando BLIP VQA: {settings.BLIP_MODEL_NAME} en {settings.BLIP_DEVICE}")
        self.device = torch.device(settings.BLIP_DEVICE)
        self.processor = BlipProcessor.from_pretrained(settings.BLIP_MODEL_NAME)
        self.model = BlipForQuestionAnswering.from_pretrained(
            settings.BLIP_MODEL_NAME
        ).to(self.device)
        self.model.eval()
        self._initialized = True
        logger.info("BLIP VQA cargado exitosamente.")

    # ── VQA ────────────────────────────────────────────────────────────────

    def answer_question(
        self,
        image: Union[Image.Image, Path, str],
        question: str,
    ) -> str:
        """
        Responde una pregunta sobre una imagen.

        Args:
            image: Imagen PIL o ruta al archivo.
            question: Pregunta en inglés sobre el contenido de la imagen.

        Returns:
            Respuesta textual del modelo.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=settings.BLIP_MAX_ANSWER_LENGTH,
            )

        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer.strip()

    def generate_image_description(
        self,
        image: Union[Image.Image, Path, str],
        questions: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Genera un diccionario de preguntas → respuestas para enriquecer
        el índice semántico de una imagen.

        Args:
            image: Imagen a describir.
            questions: Lista de preguntas (usa INDEXING_QUESTIONS por defecto).

        Returns:
            Dict {pregunta: respuesta}.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        qs = questions or INDEXING_QUESTIONS
        qa_pairs: dict[str, str] = {}

        for question in qs:
            try:
                answer = self.answer_question(image, question)
                qa_pairs[question] = answer
            except Exception as e:
                logger.warning(f"Error en VQA para pregunta '{question}': {e}")
                qa_pairs[question] = ""

        return qa_pairs

    def build_searchable_text(
        self,
        image: Union[Image.Image, Path, str],
    ) -> str:
        """
        Construye un texto concatenado con todas las respuestas VQA.
        Este texto se usa junto al embedding CLIP para mejorar la búsqueda.
        """
        qa = self.generate_image_description(image)
        parts = [f"{q.rstrip('?')}: {a}" for q, a in qa.items() if a]
        return ". ".join(parts)

    def answer_custom_question(
        self,
        image: Union[Image.Image, Path, str],
        question: str,
    ) -> str:
        """
        Responde una pregunta personalizada del usuario sobre una imagen
        (útil para el agente durante la búsqueda interactiva).
        """
        return self.answer_question(image, question)


@lru_cache(maxsize=1)
def get_blip_model() -> BLIPWrapper:
    """Factory con caché para obtener la instancia BLIP."""
    return BLIPWrapper()
