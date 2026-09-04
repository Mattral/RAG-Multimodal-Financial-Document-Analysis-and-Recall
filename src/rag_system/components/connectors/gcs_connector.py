"""Google Cloud Storage (GCS) document connector.

Guideline §4: enterprise connectors — S3, Azure Blob, GCS.

Usage:
    connector = GCSConnector(bucket="my-filings", prefix="10-K/")
    async for doc in connector.stream(tenant_id="acme"):
        await pipeline.ingest([doc.local_path], tenant_id=doc.tenant_id)

Requires: pip install google-cloud-storage
Auth: GOOGLE_APPLICATION_CREDENTIALS, gcloud ADC, or Workload Identity (GKE).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator, List, Optional

import structlog

from src.rag_system.components.connectors import BaseConnector, DiscoveredDocument

logger = structlog.get_logger(__name__)


class GCSConnector(BaseConnector):
    """GCS connector — streams PDFs directly into the ingestion pipeline."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        extensions: Optional[List[str]] = None,
        project: Optional[str] = None,
    ) -> None:
        self._bucket_name = bucket
        self._prefix = prefix
        self._extensions = extensions or [".pdf", ".docx", ".txt"]
        self._project = project

    def _check_deps(self) -> bool:
        try:
            from google.cloud import storage  # noqa: F401

            return True
        except ImportError:
            logger.warning(
                "google_cloud_storage_not_installed",
                detail="pip install google-cloud-storage",
            )
            return False

    def _get_client(self):
        from google.cloud import storage

        return storage.Client(project=self._project)

    async def list_uris(self) -> List[str]:
        if not self._check_deps():
            return []
        try:
            client = await asyncio.to_thread(self._get_client)
            blobs = await asyncio.to_thread(
                lambda: list(
                    client.list_blobs(self._bucket_name, prefix=self._prefix)
                )
            )
            return [
                f"gs://{self._bucket_name}/{b.name}"
                for b in blobs
                if any(b.name.endswith(ext) for ext in self._extensions)
                and not b.name.endswith("/")
            ]
        except Exception as exc:
            logger.error("gcs_list_failed", bucket=self._bucket_name, error=str(exc))
            return []

    async def stream(
        self, tenant_id: str = "default"
    ) -> AsyncIterator[DiscoveredDocument]:
        if not self._check_deps():
            return
        try:
            client = await asyncio.to_thread(self._get_client)
            blobs = await asyncio.to_thread(
                lambda: list(
                    client.list_blobs(self._bucket_name, prefix=self._prefix)
                )
            )
            for blob in blobs:
                if not any(blob.name.endswith(ext) for ext in self._extensions):
                    continue
                if blob.name.endswith("/"):
                    continue
                filename = Path(blob.name).name
                with tempfile.NamedTemporaryFile(
                    suffix=Path(filename).suffix, delete=False
                ) as tmp:
                    await asyncio.to_thread(
                        blob.download_to_filename, tmp.name
                    )
                    local_path = tmp.name
                yield DiscoveredDocument(
                    source_uri=f"gs://{self._bucket_name}/{blob.name}",
                    local_path=local_path,
                    filename=filename,
                    size_bytes=blob.size or 0,
                    last_modified=blob.updated.isoformat()
                    if blob.updated
                    else None,
                    tenant_id=tenant_id,
                    metadata={
                        "connector": "gcs",
                        "bucket": self._bucket_name,
                        "blob_name": blob.name,
                        "content_type": blob.content_type,
                    },
                )
                os.unlink(local_path)
        except ImportError:
            logger.warning("google_cloud_storage_not_installed")
        except Exception as exc:
            logger.error("gcs_stream_failed", error=str(exc))
