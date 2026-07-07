"""
Codebase Graph Navigator - locates SQL queries in the target repository.

Uses the pre-generated `graph.json` artifact (produced by Graphify CLI +
the SQL enrichment post-processor in the target repo's CI pipeline) to
locate SQL queries entirely in-memory.

The enriched graph contains `sql_query` nodes with:
  - "extracted_sql": the actual SQL text
  - "file": the source file path
  - "line_start" / "line_end": exact line range

This means the PR Agent can identify the target file and line range with
**zero GitHub API calls** for the lookup phase. Only one API call is
needed later to fetch the file content for patching.
"""

from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from github import Github
from github.Repository import Repository

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────


@dataclass
class SQLLocation:
    """Result of a successful SQL query lookup in the target codebase."""

    file_path: str
    line_start: int
    line_end: int
    match_context: str  # Short snippet of the matched SQL for logging
    similarity_score: float = 1.0  # 0.0-1.0 confidence in the match
    node_id: str = ""  # Graph node ID that matched

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "match_context": self.match_context,
            "similarity_score": self.similarity_score,
            "node_id": self.node_id,
        }


@dataclass
class SQLNode:
    """A parsed sql_query node from the enriched graph."""

    node_id: str
    file_path: str
    line_start: int
    line_end: int
    extracted_sql: str
    normalized_sql: str = field(init=False)

    def __post_init__(self) -> None:
        self.normalized_sql = _normalize_sql(self.extracted_sql)


# ── Normalisation ───────────────────────────────────────────────


def _normalize_sql(sql: str) -> str:
    """
    Normalize SQL text for comparison.

    Strips comments, collapses whitespace, lowercases - so that
    trivial formatting differences don't prevent matching.
    """
    # Remove SQL single-line comments
    sql = re.sub(r"--[^\n]*", "", sql)
    # Remove block comments
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    # Collapse all whitespace (newlines, tabs, multiple spaces) into single space
    sql = re.sub(r"\s+", " ", sql)
    # Strip and lowercase
    return sql.strip().lower()


# ── Navigator ───────────────────────────────────────────────────


class CodebaseGraphNavigator:
    """
    Navigate a target repo's codebase using a pre-generated graph.

    The enriched graph contains `sql_query` nodes with extracted SQL
    text, file paths, and exact line ranges. SQL matching is performed
    entirely in-memory against these nodes - no file fetching needed.
    """

    # Minimum similarity ratio (0-1) to consider a match valid.
    # 0.70 is intentionally generous to handle minor reformatting,
    # aliasing differences, and whitespace-only changes.
    SIMILARITY_THRESHOLD = 0.70

    def __init__(self, repo: Repository, token: str = "") -> None:
        self._repo = repo
        self._token = token  # Stored for artifact download (requests needs it directly)

    # ── Public API ──────────────────────────────────────────────

    def download_graph(self) -> dict:
        """
        Download the latest `graph.json` from GitHub Actions artifacts.

        Looks for an artifact named 'codebase-graph' produced by the
        generate-graph.yml workflow.

        Returns the parsed JSON dict.
        """
        logger.info(
            "Downloading codebase graph from %s artifacts...",
            self._repo.full_name,
        )

        # List artifacts - most recent first
        artifacts = self._repo.get_artifacts()

        graph_artifact = None
        for artifact in artifacts:
            if artifact.name == "codebase-graph":
                graph_artifact = artifact
                break

        if graph_artifact is None:
            raise RuntimeError(
                f"No 'codebase-graph' artifact found in {self._repo.full_name}. "
                f"Ensure the generate-graph.yml workflow has run at least once."
            )

        logger.info(
            "Found artifact '%s' (id=%s, created=%s)",
            graph_artifact.name,
            graph_artifact.id,
            graph_artifact.created_at,
        )

        # Download the artifact zip.
        # GitHub's archive_download_url always returns HTTP 302 → S3 signed URL.
        # PyGitHub's requestBlob does NOT follow redirects, so we use `requests`
        # with allow_redirects=True to correctly traverse the full redirect chain.
        import requests

        download_url = graph_artifact.archive_download_url
        response = requests.get(
            download_url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            allow_redirects=True,
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download artifact (HTTP {response.status_code}). "
                f"Check that TARGET_REPO_TOKEN has 'repo' and 'actions' read scope."
            )

        data = response.content

        # Artifact downloads are zip files containing the actual file(s)
        zip_buffer = io.BytesIO(data)
        with zipfile.ZipFile(zip_buffer) as zf:
            names = zf.namelist()
            graph_file = None
            for name in names:
                if name.endswith("graph.json"):
                    graph_file = name
                    break

            if graph_file is None:
                raise RuntimeError(
                    f"graph.json not found inside artifact zip. Files: {names}"
                )

            graph_data = json.loads(zf.read(graph_file))

        # Log summary stats
        all_nodes = graph_data.get("nodes", [])
        sql_nodes = [n for n in all_nodes if n.get("type") == "sql_query"]
        logger.info(
            "Graph loaded: %d total nodes (%d sql_query nodes), %d edges",
            len(all_nodes),
            len(sql_nodes),
            len(graph_data.get("edges", [])),
        )

        return graph_data

    def extract_sql_nodes(self, graph: dict) -> list[SQLNode]:
        """
        Extract all sql_query nodes from the enriched graph.

        These are the nodes injected by the post-processing script
        in the target repo's CI pipeline. Each contains:
          - file: source file path
          - line_start / line_end: exact line range
          - extracted_sql: the raw SQL text

        Also includes file-level graphify nodes that have a
        fallback `extracted_sql` field (e.g. migration stubs).
        """
        sql_nodes: list[SQLNode] = []

        for node in graph.get("nodes", []):
            # Primary: explicit sql_query nodes from the enrichment script
            if node.get("type") == "sql_query":
                extracted = node.get("extracted_sql", "")
                if not extracted or not extracted.strip():
                    continue

                sql_nodes.append(
                    SQLNode(
                        node_id=node["id"],
                        file_path=node["file"],
                        line_start=node["line_start"],
                        line_end=node["line_end"],
                        extracted_sql=extracted,
                    )
                )

            # Fallback: file-level graphify nodes with backfilled extracted_sql
            elif node.get("extracted_sql") and node.get("source_file"):
                extracted = node["extracted_sql"]
                if not extracted.strip():
                    continue

                sql_nodes.append(
                    SQLNode(
                        node_id=node.get("id", ""),
                        file_path=node["source_file"],
                        line_start=1,  # File-level fallback covers the whole file
                        line_end=extracted.count("\n") + 1,
                        extracted_sql=extracted,
                    )
                )

        logger.info(
            "Extracted %d SQL nodes from graph across %d files",
            len(sql_nodes),
            len({n.file_path for n in sql_nodes}),
        )

        return sql_nodes

    def locate_sql(
        self,
        original_sql: str,
        graph: dict,
    ) -> SQLLocation | None:
        """
        Locate a SQL query in the target repository using the enriched graph.

        Performs normalized text matching entirely in-memory against the
        graph's sql_query nodes. No GitHub API calls are made.

        Matching strategy (in order of priority):
        1. Exact normalized match (score = 1.0)
        2. Substring containment (score = 0.95)
        3. Fuzzy similarity above threshold (score = ratio)

        Args:
            original_sql: The SQL query text to find.
            graph: The parsed graph.json dict.

        Returns:
            SQLLocation with file_path, line_start, line_end, or None.
        """
        sql_nodes = self.extract_sql_nodes(graph)
        normalized_target = _normalize_sql(original_sql)

        if not normalized_target:
            logger.warning("Empty SQL query provided - cannot locate")
            return None

        if not sql_nodes:
            logger.warning(
                "No sql_query nodes found in graph. Ensure the target repo's "
                "CI pipeline includes the SQL enrichment post-processor."
            )
            return None

        logger.info(
            "Matching query against %d SQL nodes (target length: %d chars)",
            len(sql_nodes),
            len(normalized_target),
        )

        # ── Phase 1: Exact normalized match ─────────────────────
        for node in sql_nodes:
            if node.normalized_sql == normalized_target:
                logger.info(
                    "Exact match found: node '%s' in %s (lines %d-%d)",
                    node.node_id,
                    node.file_path,
                    node.line_start,
                    node.line_end,
                )
                return self._node_to_location(node, score=1.0)

        # ── Phase 2: Substring containment ──────────────────────
        # The target SQL might be a subset of a larger query in the
        # graph (e.g., the alert payload doesn't include CTEs or
        # trailing clauses), or vice versa.
        for node in sql_nodes:
            if normalized_target in node.normalized_sql:
                logger.info(
                    "Substring match (target in graph): node '%s' in %s",
                    node.node_id,
                    node.file_path,
                )
                return self._node_to_location(node, score=0.95)

            if node.normalized_sql in normalized_target:
                logger.info(
                    "Substring match (graph in target): node '%s' in %s",
                    node.node_id,
                    node.file_path,
                )
                return self._node_to_location(node, score=0.90)

        # ── Phase 3: Fuzzy similarity ───────────────────────────
        best_match: SQLNode | None = None
        best_score: float = 0.0

        for node in sql_nodes:
            ratio = SequenceMatcher(
                None, normalized_target, node.normalized_sql
            ).ratio()

            if ratio > best_score:
                best_score = ratio
                best_match = node

        if best_match and best_score >= self.SIMILARITY_THRESHOLD:
            logger.info(
                "Fuzzy match (%.1f%% similar): node '%s' in %s (lines %d-%d)",
                best_score * 100,
                best_match.node_id,
                best_match.file_path,
                best_match.line_start,
                best_match.line_end,
            )
            return self._node_to_location(best_match, score=best_score)

        # ── No match found ──────────────────────────────────────
        if best_match:
            logger.warning(
                "Best fuzzy match was %.1f%% similar (below %.0f%% threshold): "
                "node '%s' in %s",
                best_score * 100,
                self.SIMILARITY_THRESHOLD * 100,
                best_match.node_id,
                best_match.file_path,
            )
        else:
            logger.warning("No sql_query nodes to match against")

        return None

    # ── Private helpers ─────────────────────────────────────────

    @staticmethod
    def _node_to_location(node: SQLNode, score: float) -> SQLLocation:
        """Convert a matched SQLNode into an SQLLocation result."""
        context = node.extracted_sql.strip()
        if len(context) > 200:
            context = context[:200] + "..."

        return SQLLocation(
            file_path=node.file_path,
            line_start=node.line_start,
            line_end=node.line_end,
            match_context=context,
            similarity_score=score,
            node_id=node.node_id,
        )


# ── Factory ─────────────────────────────────────────────────────


def get_codebase_navigator(token: str, repo_name: str) -> CodebaseGraphNavigator:
    """
    Factory: create a CodebaseGraphNavigator for the given target repo.

    Args:
        token: GitHub PAT with repo + actions scope.
        repo_name: Full repo name, e.g. "SurajCSENITS/demo-tpch-app".

    Returns:
        CodebaseGraphNavigator instance.
    """
    g = Github(token)
    repo = g.get_repo(repo_name)
    return CodebaseGraphNavigator(repo, token=token)
