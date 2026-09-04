"""ColPali late-interaction visual page retriever.

Guideline §7: ColPali/ColQwen for late-interaction visual page embeddings.
Reference: Faysse et al. 2024 — https://arxiv.org/abs/2407.01449

MaxSim score: score(Q,D) = sum_i max_j cosine(q_i, d_j)
Each PDF page is embedded as N patch vectors; retrieval is over whole pages.
Requires: pip install colpali-engine torch torchvision
Graceful degradation: returns [] if colpali-engine not installed.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

from src.rag_system.components.base import BaseRetriever, RetrievedChunk

logger = structlog.get_logger(__name__)


def _maxsim(query_vecs: List[List[float]], doc_vecs: List[List[float]]) -> float:
    """MaxSim late-interaction: sum over query patches of max doc-patch similarity."""
    if not doc_vecs:
        return 0.0
    try:
        import numpy as np

        q_mat = np.array(query_vecs, dtype=np.float32)
        d_mat = np.array(doc_vecs, dtype=np.float32)
        return float(np.sum(np.max(q_mat @ d_mat.T, axis=1)))
    except ImportError:
        total = 0.0
        for qv in query_vecs:
            best = max(
                sum(a * b for a, b in zip(qv, dv, strict=True)) for dv in doc_vecs
            )
            total += best
        return total


@dataclass
class PageEmbedding:
    source_document: str
    page_number: int
    patch_embeddings: List[List[float]]
    thumbnail_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ColPaliRetriever(BaseRetriever):
    """ColQwen2 late-interaction visual page retriever with index persistence."""

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        device: str = "auto",
        index_path: Optional[str] = "./data/colpali_index.json",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._index_path = Path(index_path) if index_path else None
        self._model = None
        self._processor = None
        self._index: List[PageEmbedding] = []
        self._loaded = False

    @property
    def name(self) -> str:
        return f"colpali/{self._model_name}"

    def _has_deps(self) -> bool:
        try:
            import colpali_engine  # noqa: F401

            return True
        except ImportError:
            logger.warning(
                "colpali_not_installed",
                detail="pip install colpali-engine torch torchvision",
            )
            return False

    def _load_model(self) -> bool:
        if self._loaded:
            return True
        if not self._has_deps():
            return False
        try:
            import torch
            from colpali_engine.models import ColQwen2, ColQwen2Processor

            device = self._device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = ColQwen2Processor.from_pretrained(self._model_name)
            self._model = ColQwen2.from_pretrained(
                self._model_name,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map=device,
            ).eval()
            self._loaded = True
            logger.info(
                "colpali_model_loaded", model=self._model_name, device=device
            )
            return True
        except Exception as exc:
            logger.error("colpali_model_failed", error=str(exc))
            return False

    def _embed_image_sync(self, image_path: str) -> List[List[float]]:
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        inputs = self._processor.process_images([img])
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        vecs = out.last_hidden_state[0]
        vecs = vecs / vecs.norm(dim=-1, keepdim=True)
        return vecs.cpu().float().tolist()

    def _embed_query_sync(self, query: str) -> List[List[float]]:
        import torch

        inputs = self._processor.process_queries([query])
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        vecs = out.last_hidden_state[0]
        vecs = vecs / vecs.norm(dim=-1, keepdim=True)
        return vecs.cpu().float().tolist()

    async def build_index(
        self, image_paths: List[str], source_document: str
    ) -> List[str]:
        steps = []
        if not await asyncio.to_thread(self._load_model):
            steps.append("ColPali unavailable — page images not indexed visually")
            return steps
        steps.append(
            f"ColPali: embedding {len(image_paths)} pages with {self._model_name}..."
        )
        for i, img_path in enumerate(image_paths):
            try:
                vecs = await asyncio.to_thread(self._embed_image_sync, img_path)
                self._index.append(
                    PageEmbedding(
                        source_document=source_document,
                        page_number=i + 1,
                        patch_embeddings=vecs,
                        thumbnail_path=img_path,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "colpali_embed_page_failed", page=i + 1, error=str(exc)
                )
        steps.append(f"ColPali: indexed {len(image_paths)} pages")
        if self._index_path:
            await asyncio.to_thread(self._save_index)
        return steps

    def _save_index(self) -> None:
        if not self._index_path:
            return
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "source_document": e.source_document,
                "page_number": e.page_number,
                "patch_embeddings": e.patch_embeddings,
                "thumbnail_path": e.thumbnail_path,
                "metadata": e.metadata,
            }
            for e in self._index
        ]
        self._index_path.write_text(json.dumps(data))
        logger.info("colpali_index_saved", num_pages=len(data))

    def load_index(self, path: Optional[str] = None) -> bool:
        p = Path(path) if path else self._index_path
        if not p or not p.exists():
            return False
        try:
            self._index = [PageEmbedding(**d) for d in json.loads(p.read_text())]
            logger.info("colpali_index_loaded", num_pages=len(self._index))
            return True
        except Exception as exc:
            logger.error("colpali_index_load_failed", error=str(exc))
            return False

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        if not self._index:
            self.load_index()
        if not self._index:
            return []
        if not await asyncio.to_thread(self._load_model):
            return []
        try:
            qvecs = await asyncio.to_thread(self._embed_query_sync, query)
        except Exception as exc:
            logger.error("colpali_query_embed_failed", error=str(exc))
            return []

        scored: List[Tuple[float, PageEmbedding]] = []
        for pe in self._index:
            if (
                filters
                and "source_document" in filters
                and pe.source_document != filters["source_document"]
            ):
                continue
            scored.append((_maxsim(qvecs, pe.patch_embeddings), pe))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(
                text=(
                    f"[Visual page — {pe.source_document}, p.{pe.page_number}]"
                    f"\nColPali MaxSim: {score:.4f}"
                ),
                score=score,
                source_document=pe.source_document,
                page_number=pe.page_number,
                metadata={
                    "method": "colpali_maxsim",
                    "model": self._model_name,
                    "thumbnail_path": pe.thumbnail_path,
                },
            )
            for score, pe in scored[:top_k]
        ]
