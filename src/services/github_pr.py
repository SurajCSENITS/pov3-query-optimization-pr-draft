"""
GitHub PR Creator — creates real Draft Pull Requests in the target repository.

Uses PyGitHub to:
1. Create a new branch from the default branch
2. Commit the patched file via the GitHub Contents API (no clone)
3. Open a Draft Pull Request with the AI-generated body

This service is called by the PR Agent after the codebase graph navigator
has identified the file to patch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from github import Github, GithubException
from github.Repository import Repository

logger = logging.getLogger(__name__)


@dataclass
class PRResult:
    """Result of a successful Draft PR creation."""

    pr_number: int
    pr_url: str
    branch_name: str
    commit_sha: str
    file_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pr_number": self.pr_number,
            "pr_url": self.pr_url,
            "branch_name": self.branch_name,
            "commit_sha": self.commit_sha,
            "file_path": self.file_path,
        }


class GitHubPRCreator:
    """
    Creates Draft PRs in the target application repository.

    All file operations use the GitHub Contents API, avoiding the
    need to clone the repository locally.
    """

    def __init__(self, repo: Repository) -> None:
        self._repo = repo

    # ── Public API ──────────────────────────────────────────────

    def create_draft_pr(
        self,
        *,
        file_path: str,
        patched_content: str,
        branch_name: str,
        pr_title: str,
        pr_body: str,
        commit_message: str,
        base_branch: str = "main",
        labels: list[str] | None = None,
    ) -> PRResult:
        """
        Create a complete Draft PR: branch → commit → pull request.

        Args:
            file_path: Path of the file to modify (relative to repo root).
            patched_content: The full new content of the file after patching.
            branch_name: Name for the new feature branch.
            pr_title: Title for the pull request.
            pr_body: Markdown body for the pull request.
            commit_message: Git commit message.
            base_branch: Branch to base the PR on (default: main).
            labels: Optional labels to add to the PR.

        Returns:
            PRResult with the PR URL, number, branch, and commit SHA.

        Raises:
            GithubException: If any GitHub API call fails.
            RuntimeError: If the file doesn't exist in the target repo.
        """
        # Step 1: Create branch
        logger.info("Creating branch '%s' from '%s'…", branch_name, base_branch)
        self._create_branch(branch_name, base_branch)

        # Step 2: Commit the patched file
        logger.info("Committing patched file '%s'…", file_path)
        commit_sha = self._commit_file(
            file_path=file_path,
            new_content=patched_content,
            branch_name=branch_name,
            base_branch=base_branch,
            commit_message=commit_message,
        )

        # Step 3: Open Draft PR
        logger.info("Opening Draft PR: %s", pr_title)
        pr = self._repo.create_pull(
            title=pr_title,
            body=pr_body,
            head=branch_name,
            base=base_branch,
            draft=True,
        )

        logger.info("✅ Draft PR #%d created: %s", pr.number, pr.html_url)

        # Step 4: Add labels (best-effort — don't fail the PR if labels fail)
        if labels:
            try:
                pr.add_to_labels(*labels)
                logger.info("Labels added: %s", labels)
            except GithubException as exc:
                logger.warning("Failed to add labels (non-fatal): %s", exc)

        return PRResult(
            pr_number=pr.number,
            pr_url=pr.html_url,
            branch_name=branch_name,
            commit_sha=commit_sha,
            file_path=file_path,
        )

    # ── Private helpers ─────────────────────────────────────────

    def _create_branch(self, branch_name: str, base_branch: str) -> None:
        """Create a new branch from the tip of base_branch."""
        try:
            source = self._repo.get_branch(base_branch)
        except GithubException:
            raise RuntimeError(
                f"Base branch '{base_branch}' not found in {self._repo.full_name}"
            )

        ref = f"refs/heads/{branch_name}"

        try:
            self._repo.create_git_ref(ref=ref, sha=source.commit.sha)
        except GithubException as exc:
            if exc.status == 422:
                # Branch already exists — might be a retry. Log and continue.
                logger.warning(
                    "Branch '%s' already exists — reusing it", branch_name
                )
            else:
                raise

    def _commit_file(
        self,
        *,
        file_path: str,
        new_content: str,
        branch_name: str,
        base_branch: str,
        commit_message: str,
    ) -> str:
        """
        Commit a modified file to the branch.

        Uses the GitHub Contents API (update_file), which handles
        the blob + tree + commit creation automatically.

        Returns the commit SHA.
        """
        # Get the current file content + SHA from the base branch
        try:
            contents = self._repo.get_contents(file_path, ref=base_branch)
        except GithubException as exc:
            if exc.status == 404:
                raise RuntimeError(
                    f"File '{file_path}' not found in {self._repo.full_name}@{base_branch}"
                )
            raise

        if isinstance(contents, list):
            raise RuntimeError(f"'{file_path}' is a directory, not a file")

        result = self._repo.update_file(
            path=file_path,
            message=commit_message,
            content=new_content,
            sha=contents.sha,
            branch=branch_name,
        )

        commit_sha = result["commit"].sha
        logger.info("Committed %s → %s", file_path, commit_sha[:8])
        return commit_sha


def patch_file_content(
    original_content: str,
    original_sql: str,
    optimized_sql: str,
    line_start: int,
    line_end: int,
) -> str:
    """
    Patch a file by replacing a SQL query block with its optimized version.

    Uses the line range from the codebase graph to scope the replacement,
    ensuring only the targeted query is modified.

    Args:
        original_content: Full original file content.
        original_sql: The SQL query text to replace.
        optimized_sql: The optimized SQL to insert.
        line_start: 1-indexed start line of the match.
        line_end: 1-indexed end line of the match.

    Returns:
        The patched file content.
    """
    lines = original_content.splitlines(keepends=True)

    # Extract the region identified by the graph navigator
    region_start = line_start - 1  # Convert to 0-indexed
    region_end = line_end  # Exclusive end for slicing

    region_text = "".join(lines[region_start:region_end])

    # Perform the replacement within the identified region
    # Use case-insensitive replacement to handle minor formatting differences
    import re

    # Build a pattern that matches the original SQL with flexible whitespace
    escaped = re.escape(original_sql)
    # Allow flexible whitespace matching
    flexible_pattern = re.sub(r"\\\s+", r"\\s+", escaped)

    patched_region = re.sub(
        flexible_pattern, optimized_sql, region_text, count=1, flags=re.IGNORECASE
    )

    if patched_region == region_text:
        # Regex didn't match — try a simpler direct replacement
        # This handles cases where the SQL is a simple substring
        if original_sql.strip() in region_text:
            patched_region = region_text.replace(
                original_sql.strip(), optimized_sql.strip(), 1
            )
        else:
            logger.warning(
                "Could not find exact SQL match in region (lines %d-%d). "
                "Falling back to full region replacement.",
                line_start,
                line_end,
            )
            # Last resort: replace the entire region content
            patched_region = region_text.replace(
                region_text.strip(), optimized_sql.strip(), 1
            )

    # Reassemble the file
    patched_lines = lines[:region_start] + [patched_region] + lines[region_end:]
    return "".join(patched_lines)


def get_pr_creator(token: str, repo_name: str) -> GitHubPRCreator:
    """
    Factory: create a GitHubPRCreator for the given target repo.

    Args:
        token: GitHub PAT with repo scope on the target.
        repo_name: Full repo name, e.g. "SurajCSENITS/demo-tpch-app".

    Returns:
        GitHubPRCreator instance.
    """
    g = Github(token)
    repo = g.get_repo(repo_name)
    return GitHubPRCreator(repo)
