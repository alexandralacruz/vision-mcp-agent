# repository/manager.py
"""
Gestor del repositorio de imágenes.
Administra el almacenamiento, indexado y búsqueda de imágenes.
"""
from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from config.settings import settings
from models.clip_model import get_clip_model
from models.blip_model import get_blip_model

logger = logging.getLogger(__name__)


class ImageMetadata:
    """Metadatos de una imagen indexada."""

    def __init__(
        self,
        image_id: str,
        filename: str,
        original_path: str,
        vqa_description: dict[str, str],
        searchable_text: str,
        indexed_at: str,
        file_size: int,
        width: int,
        height: int,
        tags: list[str] | None = None,
    ):
        self.image_id = image_id
        self.filename = filename
        self.original_path = original_path
        self.vqa_description = vqa_description
        self.searchable_text = searchable_text
        self.indexed_at = indexed_at
        self.file_size = file_size
        self.width = width
        self.height = height
        self.tags = tags or []

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, data: dict) -> "ImageMetadata":
        return cls(**data)


class RepositoryManager:
    """Gestor principal del repositorio de imágenes con embeddings."""

    def __init__(self):
        self.index_path = settings.INDEX_PATH
        self.images_path = settings.REPOSITORY_PATH
        self.embeddings_path = settings.EMBEDDINGS_PATH

        # Crear directorios
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.embeddings_path.mkdir(parents=True, exist_ok=True)

        # Cargar índice existente
        self._index: dict[str, ImageMetadata] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._load_index()

    # ── Índice ──────────────────────────────────────────────────────────────

    def _load_index(self):
        """Carga el índice de metadatos y embeddings del disco."""
        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._index = {
                img_id: ImageMetadata.from_dict(meta)
                for img_id, meta in raw.items()
            }
            logger.info(f"Índice cargado: {len(self._index)} imágenes.")

        # Cargar embeddings
        for img_id in self._index:
            emb_file = self.embeddings_path / f"{img_id}.npy"
            if emb_file.exists():
                self._embeddings[img_id] = np.load(str(emb_file))

    def _save_index(self):
        """Persiste el índice de metadatos en disco."""
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v.to_dict() for k, v in self._index.items()},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _save_embedding(self, image_id: str, embedding: np.ndarray):
        """Persiste el embedding de una imagen."""
        np.save(str(self.embeddings_path / f"{image_id}.npy"), embedding)

    # ── Agregar imagen ──────────────────────────────────────────────────────

    def add_image(
        self,
        image_path: Path | str,
        tags: list[str] | None = None,
    ) -> ImageMetadata:
        """
        Indexa una imagen nueva:
        1. Copia la imagen al repositorio.
        2. Genera embedding CLIP.
        3. Genera descripción VQA con BLIP.
        4. Guarda metadatos y embedding.

        Returns:
            ImageMetadata del registro creado.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

        suffix = image_path.suffix.lower()
        if suffix not in settings.SUPPORTED_FORMATS:
            raise ValueError(f"Formato no soportado: {suffix}")

        image_id = str(uuid.uuid4())
        dest_filename = f"{image_id}{suffix}"
        dest_path = self.images_path / dest_filename

        # Copiar imagen al repositorio
        shutil.copy2(image_path, dest_path)

        # Abrir imagen
        img = Image.open(dest_path).convert("RGB")
        width, height = img.size

        logger.info(f"Generando embedding CLIP para {image_id}...")
        clip = get_clip_model()
        embedding = clip.embed_image(img)

        logger.info(f"Generando descripción VQA BLIP para {image_id}...")
        blip = get_blip_model()
        vqa_description = blip.generate_image_description(img)
        searchable_text = blip.build_searchable_text(img)

        # Crear metadatos
        meta = ImageMetadata(
            image_id=image_id,
            filename=dest_filename,
            original_path=str(image_path),
            vqa_description=vqa_description,
            searchable_text=searchable_text,
            indexed_at=datetime.utcnow().isoformat(),
            file_size=dest_path.stat().st_size,
            width=width,
            height=height,
            tags=tags or [],
        )

        # Guardar
        self._index[image_id] = meta
        self._embeddings[image_id] = embedding
        self._save_embedding(image_id, embedding)
        self._save_index()

        logger.info(f"Imagen indexada: {image_id}")
        return meta

    # ── Búsqueda ────────────────────────────────────────────────────────────

    def search_by_text(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        """
        Busca imágenes por descripción textual usando embeddings CLIP.

        Returns:
            Lista de {image_id, score, metadata, image_path} ordenada por relevancia.
        """
        if not self._embeddings:
            return []

        top_k = top_k or settings.MAX_RESULTS
        threshold = threshold or settings.CLIP_SIMILARITY_THRESHOLD

        clip = get_clip_model()
        ranked = clip.rank_images_by_text(query, self._embeddings)

        results = []
        for image_id, score in ranked[:top_k]:
            if score < threshold:
                continue
            meta = self._index.get(image_id)
            if meta:
                results.append({
                    "image_id": image_id,
                    "score": round(score, 4),
                    "metadata": meta.to_dict(),
                    "image_path": str(self.images_path / meta.filename),
                })

        return results

    def search_by_image(
        self,
        query_image_path: Path | str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        """Busca imágenes visualmente similares a una imagen de consulta."""
        if not self._embeddings:
            return []

        top_k = top_k or settings.MAX_RESULTS
        threshold = threshold or settings.CLIP_SIMILARITY_THRESHOLD

        clip = get_clip_model()
        ranked = clip.rank_images_by_image(query_image_path, self._embeddings)

        results = []
        for image_id, score in ranked[:top_k]:
            if score < threshold:
                continue
            meta = self._index.get(image_id)
            if meta:
                results.append({
                    "image_id": image_id,
                    "score": round(score, 4),
                    "metadata": meta.to_dict(),
                    "image_path": str(self.images_path / meta.filename),
                })

        return results

    # ── CRUD básico ─────────────────────────────────────────────────────────

    def get_image(self, image_id: str) -> ImageMetadata | None:
        return self._index.get(image_id)

    def delete_image(self, image_id: str) -> bool:
        if image_id not in self._index:
            return False
        meta = self._index.pop(image_id)
        self._embeddings.pop(image_id, None)

        # Eliminar archivos
        img_file = self.images_path / meta.filename
        if img_file.exists():
            img_file.unlink()
        emb_file = self.embeddings_path / f"{image_id}.npy"
        if emb_file.exists():
            emb_file.unlink()

        self._save_index()
        return True

    def list_images(self, limit: int = 50, offset: int = 0) -> list[dict]:
        items = list(self._index.values())[offset: offset + limit]
        return [
            {
                **m.to_dict(),
                "image_path": str(self.images_path / m.filename),
            }
            for m in items
        ]

    @property
    def total_images(self) -> int:
        return len(self._index)


# Singleton
_repository: RepositoryManager | None = None


def get_repository() -> RepositoryManager:
    global _repository
    if _repository is None:
        _repository = RepositoryManager()
    return _repository
