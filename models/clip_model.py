# models/clip_model.py
"""
Wrapper para CLIP (Contrastive Language-Image Pre-Training).
Genera embeddings de imágenes y texto en el mismo espacio vectorial,
permitiendo búsqueda semántica cruzada.
"""
from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import Union
from functools import lru_cache

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from config.settings import settings

logger = logging.getLogger(__name__)


class CLIPWrapper:
    """Singleton wrapper para el modelo CLIP."""

    _instance: "CLIPWrapper | None" = None

    def __new__(cls) -> "CLIPWrapper":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        logger.info(f"Cargando CLIP: {settings.CLIP_MODEL_NAME} en {settings.CLIP_DEVICE}")
        self.device = torch.device(settings.CLIP_DEVICE)
        self.model = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
        self.model.eval()
        self._initialized = True
        logger.info("CLIP cargado exitosamente.")

    # ── Embeddings de imagen ────────────────────────────────────────────────

    def embed_image(self, image: Union[Image.Image, Path, str]) -> np.ndarray:
        """
        Genera un embedding normalizado para una imagen.
        Retorna vector float32 de dimensión CLIP_EMBEDDING_DIM.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            features = self.model.get_image_features(**inputs).pooler_output
            features = features / features.norm(dim=-1, keepdim=True)  # normalizar

        return features.cpu().numpy().astype(np.float32).flatten()

    # ── Embeddings de texto ─────────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """
        Genera un embedding normalizado para una descripción textual.
        Compatible con el espacio de embeddings de imagen.
        """
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        with torch.no_grad():
            features = self.model.get_text_features(**inputs).pooler_output
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype(np.float32).flatten()

    # ── Similitud ───────────────────────────────────────────────────────────

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Similitud coseno entre dos vectores ya normalizados."""
        return float(np.dot(vec_a, vec_b))

    def rank_images_by_text(
        self,
        text_query: str,
        image_embeddings: dict[str, np.ndarray],
    ) -> list[tuple[str, float]]:
        """
        Rankea imágenes por similitud con un query textual.

        Args:
            text_query: Descripción en lenguaje natural.
            image_embeddings: {image_id: embedding_vector}

        Returns:
            Lista de (image_id, score) ordenada de mayor a menor similitud.
        """
        text_emb = self.embed_text(text_query)
        scores = [
            (img_id, self.cosine_similarity(text_emb, img_emb))
            for img_id, img_emb in image_embeddings.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def rank_images_by_image(
        self,
        query_image: Union[Image.Image, Path, str],
        image_embeddings: dict[str, np.ndarray],
    ) -> list[tuple[str, float]]:
        """Rankea imágenes por similitud visual con otra imagen de consulta."""
        query_emb = self.embed_image(query_image)
        scores = [
            (img_id, self.cosine_similarity(query_emb, img_emb))
            for img_id, img_emb in image_embeddings.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)


@lru_cache(maxsize=1)
def get_clip_model() -> CLIPWrapper:
    """Factory con caché para obtener la instancia CLIP."""
    return CLIPWrapper()
