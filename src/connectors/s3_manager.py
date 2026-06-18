"""
S3 Manager — uploads and retrieves optimization reports from S3.

Handles:
- Uploading structured JSON reports to the RAG source bucket
- Generating consistent S3 key paths
- Triggering Bedrock Knowledge Base sync after upload

Reports stored in S3 are ingested by Bedrock Knowledge Bases
and used as the RAG context for future optimizations.

Usage:
    from src.connectors.s3_manager import get_s3_manager
    manager = get_s3_manager()
    key = manager.upload_report(report_id="abc123", data={...})
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class S3Manager:
    """
    Singleton wrapper around boto3 S3 client for report storage.
    """

    _instance: S3Manager | None = None
    _client: Any = None

    def __new__(cls) -> S3Manager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._settings = get_settings()

    def _get_client(self) -> Any:
        """Lazily create and return the boto3 S3 client."""
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                region_name=self._settings.aws_region,
                aws_access_key_id=self._settings.aws_access_key_id,
                aws_secret_access_key=self._settings.aws_secret_access_key,
            )
            logger.info("S3 client created (region=%s)", self._settings.aws_region)
        return self._client

    # ── Key construction ───────────────────────────────────────────────────────

    def _make_key(self, report_id: str, section: str = "full") -> str:
        """
        Build a consistent S3 key for a report section.

        Key format: reports/{report_id}/{section}.json

        Sections:
            full    — the complete report JSON (default)
            summary — metadata + bottleneck types only
            sql     — original + optimized SQL pair
        """
        prefix = self._settings.s3_reports_prefix.rstrip("/")
        return f"{prefix}/{report_id}/{section}.json"

    # ── Upload ─────────────────────────────────────────────────────────────────

    def upload_report(self, report_id: str, data: dict[str, Any]) -> str:
        """
        Upload a full optimization report JSON to S3.

        Also writes a trimmed 'summary' object for faster RAG retrieval.

        Args:
            report_id: Unique identifier for this optimization run.
            data: The complete report dictionary.

        Returns:
            The S3 key of the uploaded full report.

        Raises:
            RuntimeError: If the upload fails.
        """
        client = self._get_client()
        bucket = self._settings.s3_bucket_name

        # ── Full report ──────────────────────────────────────────
        full_key = self._make_key(report_id, "full")
        try:
            client.put_object(
                Bucket=bucket,
                Key=full_key,
                Body=json.dumps(data, indent=2, default=str),
                ContentType="application/json",
            )
            logger.info("Uploaded full report to s3://%s/%s", bucket, full_key)
        except Exception as e:
            logger.error("S3 upload failed for %s: %s", full_key, e)
            raise RuntimeError(f"S3 upload failed: {e}") from e

        # ── Summary chunk (optimized for RAG retrieval) ──────────
        summary = {
            "report_id": report_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_id": data.get("query_id", ""),
            "bottleneck_types": data.get("bottleneck_types", []),
            "optimizations_applied": data.get("optimizations_applied", []),
            "root_cause": data.get("root_cause", ""),
            "performance": data.get("performance", {}),
            "confidence_score": data.get("confidence_score", 0.0),
            "validated": data.get("validated", False),
            "original_sql": data.get("original_sql", ""),
            "optimized_sql": data.get("optimized_sql", ""),
        }
        summary_key = self._make_key(report_id, "summary")
        try:
            client.put_object(
                Bucket=bucket,
                Key=summary_key,
                Body=json.dumps(summary, indent=2, default=str),
                ContentType="application/json",
            )
            logger.info("Uploaded summary chunk to s3://%s/%s", bucket, summary_key)
        except Exception as e:
            logger.warning("Could not upload summary chunk to S3: %s", e)
            # Non-fatal — full report already uploaded

        # ── Trigger Bedrock Sync ─────────────────────────────────
        if self._settings.bedrock_kb_id and self._settings.bedrock_data_source_id:
            self.sync_knowledge_base()

        return full_key

    def report_exists(self, report_id: str) -> bool:
        """Check if a report already exists in S3 (idempotency guard)."""
        client = self._get_client()
        bucket = self._settings.s3_bucket_name
        key = self._make_key(report_id, "full")
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def sync_knowledge_base(self) -> None:
        """Trigger a sync job for the Bedrock Knowledge Base Data Source."""
        try:
            import boto3

            bedrock_client = boto3.client(
                "bedrock-agent",
                region_name=self._settings.aws_region,
                aws_access_key_id=self._settings.aws_access_key_id,
                aws_secret_access_key=self._settings.aws_secret_access_key,
            )
            
            response = bedrock_client.start_ingestion_job(
                knowledgeBaseId=self._settings.bedrock_kb_id,
                dataSourceId=self._settings.bedrock_data_source_id,
            )
            job_id = response.get("ingestionJob", {}).get("ingestionJobId", "unknown")
            logger.info("Triggered Bedrock KB sync job: %s", job_id)
        except Exception as e:
            logger.error("Failed to trigger Bedrock KB sync: %s", e)


def get_s3_manager() -> S3Manager:
    """Return the process-wide singleton S3Manager."""
    return S3Manager()
