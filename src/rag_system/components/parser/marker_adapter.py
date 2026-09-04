"""Marker PDF parser adapter.

Guideline §4: 'Marker — high-quality PDF-to-markdown conversion preserving
tables, headers, and math. Best for dense academic and financial documents.'

GitHub: https://github.com/VikParuchuri/marker

Startup:
    pip install marker-pdf

Features over unstructured:
- Superior table fidelity (complex multi-column tables)
- Preserves mathematical notation (formulas in financial footnotes)
- Better layout reading order (complex multi-column PDFs)
- Outputs clean Markdown suitable for chunking

Usage:
    VECTOR_STORE_CONFIG__PDF_PARSING_PROVIDER=marker
    or programmatically:
        parser = MarkerParser()
        elements = await parser.parse("10k.pdf")
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional

import structlog

from src.rag_system.components.base import BaseParser, DocumentElement

logger = structlog.get_logger(__name__)


class MarkerParser(BaseParser):
    """Marker PDF-to-Markdown parser with chunking.

    Produces high-fidelity markdown output then applies sentence-aware
    chunking to avoid splitting mid-thought.
    """

    def __init__(self, chunk_size: int = 3800, chunk_overlap: int = 200) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def name(self) -> str:
        return "marker"

    def _check_marker(self) -> bool:
        try:
            import marker  # noqa: F401

            return True
        except ImportError:
            logger.warning(
                "marker_not_installed",
                detail="pip install marker-pdf",
                fallback="UnstructuredParser will be used",
            )
            return False

    def _parse_sync(self, file_path: str, tenant_id: Optional[str]) -> List[DocumentElement]:
        if not self._check_marker():
            # Graceful fallback to unstructured
            from src.rag_system.components.parser import UnstructuredParser

            return asyncio.get_event_loop().run_until_complete(
                UnstructuredParser().parse(file_path, tenant_id=tenant_id)
            )

        try:
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models

            models = load_all_models()
            full_text, images, metadata = convert_single_pdf(file_path, models)

            source = Path(file_path).name
            now = datetime.now(UTC).isoformat()
            elements: List[DocumentElement] = []

            # Split on double newlines (paragraph boundaries)
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            chunk_buffer = ""
            page_counter = 1

            for para in paragraphs:
                if len(chunk_buffer) + len(para) < self._chunk_size:
                    chunk_buffer += "\n\n" + para
                else:
                    if chunk_buffer.strip():
                        text = chunk_buffer.strip()
                        etype = "table" if text.startswith("|") else "text"
                        elements.append(
                            DocumentElement(
                                type=etype,
                                text=text,
                                source_document=source,
                                page_number=page_counter,
                                ingest_timestamp=now,
                                content_hash=hashlib.sha256(
                                    text.encode()
                                ).hexdigest()[:12],
                                tenant_id=tenant_id,
                                metadata={"parser": self.name},
                            )
                        )
                        # Overlap: carry last N chars into next chunk
                        chunk_buffer = (
                            chunk_buffer[-self._chunk_overlap :]
                            + "\n\n"
                            + para
                        )
                        page_counter += 1
                    else:
                        chunk_buffer = para

            # Final chunk
            if chunk_buffer.strip():
                text = chunk_buffer.strip()
                etype = "table" if text.startswith("|") else "text"
                elements.append(
                    DocumentElement(
                        type=etype,
                        text=text,
                        source_document=source,
                        page_number=page_counter,
                        ingest_timestamp=now,
                        content_hash=hashlib.sha256(text.encode()).hexdigest()[:12],
                        tenant_id=tenant_id,
                        metadata={"parser": self.name},
                    )
                )

            logger.info(
                "marker_parse_complete",
                file=source,
                num_chunks=len(elements),
                total_chars=len(full_text),
            )
            return elements

        except Exception as exc:
            logger.error("marker_parse_failed", file=file_path, error=str(exc))
            return []

    async def parse(
        self, file_path: str, tenant_id: Optional[str] = None
    ) -> List[DocumentElement]:
        return await asyncio.to_thread(self._parse_sync, file_path, tenant_id)

    async def parse_batch(
        self, file_paths: List[str], tenant_id: Optional[str] = None
    ) -> List[DocumentElement]:
        tasks = [self.parse(fp, tenant_id=tenant_id) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elements: List[DocumentElement] = []
        for fp, result in zip(file_paths, results, strict=True):
            if isinstance(result, Exception):
                logger.error("marker_batch_file_failed", file=fp, error=str(result))
            else:
                elements.extend(result)
        return elements
