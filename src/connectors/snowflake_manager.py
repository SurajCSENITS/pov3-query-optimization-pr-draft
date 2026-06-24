"""
Snowflake Connection Manager.

Provides a reusable, singleton-pattern connection manager for
executing queries against Snowflake. Includes:

- Connection pooling via a cached connector instance
- Health check method
- Query execution helper (returns list of dicts)
- Convenience methods for EXPLAIN and QUERY_HISTORY
- Graceful error handling with retries on transient failures
- Context manager support for clean resource cleanup

Usage:
    from src.connectors.snowflake_manager import SnowflakeConnectionManager

    manager = SnowflakeConnectionManager()
    if manager.health_check():
        results = manager.execute_query("SELECT CURRENT_VERSION()")
"""

from __future__ import annotations

import logging
import time
from typing import Any

import snowflake.connector
from snowflake.connector import DictCursor
from snowflake.connector.errors import (
    DatabaseError,
    InterfaceError,
    OperationalError,
    ProgrammingError,
)

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Transient error codes that are safe to retry
_TRANSIENT_ERROR_CODES = {
    "250001",  # Could not connect to Snowflake
    "250003",  # Connection refused
    "250006",  # Network error
}

_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2


class SnowflakeConnectionManager:
    """
    Singleton-pattern Snowflake connection manager.

    Maintains a single connection instance per manager object.
    Use `get_connection_manager()` for a process-wide singleton.
    """

    _instance: SnowflakeConnectionManager | None = None
    _connection: snowflake.connector.SnowflakeConnection | None = None

    def __new__(cls) -> SnowflakeConnectionManager:
        """Ensure only one instance exists per process (singleton)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── Connection lifecycle ────────────────────────────────────

    def _get_connection(self) -> snowflake.connector.SnowflakeConnection:
        """
        Return the active connection, creating one if needed.

        Reconnects automatically if the previous connection was closed.
        """
        if self._connection is None or self._connection.is_closed():
            logger.info(
                "Connecting to Snowflake account=%s warehouse=%s",
                self._settings.snowflake_account,
                self._settings.snowflake_warehouse,
            )
            self._connection = snowflake.connector.connect(
                account=self._settings.snowflake_account,
                user=self._settings.snowflake_user,
                password=self._settings.snowflake_password,
                warehouse=self._settings.snowflake_warehouse,
                database=self._settings.snowflake_database,
                schema=self._settings.snowflake_schema,
                role=self._settings.snowflake_role,
                # Timeout settings for production stability
                login_timeout=30,
                network_timeout=60,
                client_session_keep_alive=True,
            )
            logger.info("Snowflake connection established successfully.")
        return self._connection

    def close(self) -> None:
        """Close the active connection if one exists."""
        if self._connection and not self._connection.is_closed():
            self._connection.close()
            logger.info("Snowflake connection closed.")
        self._connection = None

    # ── Context manager support ─────────────────────────────────

    def __enter__(self) -> SnowflakeConnectionManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── Health check ────────────────────────────────────────────

    def health_check(self) -> bool:
        """
        Verify connectivity by running a lightweight query.

        Returns True if the connection is healthy, False otherwise.
        """
        try:
            result = self.execute_query("SELECT CURRENT_VERSION() AS version")
            version = result[0]["VERSION"] if result else "unknown"
            logger.info("Snowflake health check passed. Version: %s", version)
            return True
        except Exception as e:
            logger.warning("Snowflake health check failed: %s", e)
            return False

    # ── Query execution ─────────────────────────────────────────

    def execute_query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        fetch_results: bool = True,
        return_sfqid: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
        """
        Execute a SQL query and return results as a list of dicts.

        Includes retry logic for transient connection failures.
        Uses DictCursor so each row is a dictionary keyed by column name.

        Args:
            sql: The SQL statement to execute.
            params: Optional parameter bindings for the query.

        Returns:
            List of dictionaries, one per result row.

        Raises:
            DatabaseError: If the query fails after all retries.
        """
        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                conn = self._get_connection()
                cursor = conn.cursor(DictCursor)
                cursor.execute(sql, params or {})
                sfqid = getattr(cursor, "sfqid", "")
                if fetch_results:
                    results = cursor.fetchall()
                else:
                    results = []
                cursor.close()
                if return_sfqid:
                    return results, sfqid
                return results

            except (OperationalError, InterfaceError) as e:
                last_error = e
                error_code = str(getattr(e, "errno", ""))

                if error_code in _TRANSIENT_ERROR_CODES and attempt < _MAX_RETRIES:
                    wait = _RETRY_BACKOFF_SECONDS * attempt
                    logger.warning(
                        "Transient Snowflake error (attempt %d/%d): %s. "
                        "Retrying in %ds...",
                        attempt,
                        _MAX_RETRIES,
                        e,
                        wait,
                    )
                    self.close()  # Force reconnect on next attempt
                    time.sleep(wait)
                else:
                    logger.error("Snowflake query failed: %s", e)
                    raise

            except ProgrammingError as e:
                # SQL syntax errors, permission issues — do not retry
                logger.error("Snowflake SQL error: %s\nQuery: %s", e, sql)
                raise

        # Should not reach here, but safety net
        raise DatabaseError(f"Query failed after {_MAX_RETRIES} attempts: {last_error}")

    # ── Convenience methods ─────────────────────────────────────

    def explain_query(self, sql: str) -> list[dict[str, Any]]:
        """
        Run EXPLAIN on a SQL query and return the execution plan.

        Args:
            sql: The SQL query to explain (without the EXPLAIN prefix).

        Returns:
            List of plan step dictionaries.
        """
        explain_sql = f"EXPLAIN USING TEXT {sql}"
        logger.info("Running EXPLAIN for query: %.100s...", sql)
        return self.execute_query(explain_sql)

    def get_query_history(
        self,
        query_id: str | None = None,
        query_text_fragment: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Fetch query execution history from ACCOUNT_USAGE.

        Can filter by exact query_id or by a SQL text fragment.
        Note: ACCOUNT_USAGE has ~45 min latency.

        Args:
            query_id: Exact Snowflake query ID to look up.
            query_text_fragment: Substring to search in query text.
            limit: Max number of results.

        Returns:
            List of query history records.
        """
        conditions = ["1=1"]
        if query_id:
            conditions.append(f"QUERY_ID = '{query_id}'")
        if query_text_fragment:
            # Escape single quotes in the fragment
            safe_fragment = query_text_fragment.replace("'", "''")
            conditions.append(f"QUERY_TEXT ILIKE '%{safe_fragment}%'")

        where_clause = " AND ".join(conditions)

        sql = f"""
        SELECT
            QUERY_ID,
            QUERY_TEXT,
            DATABASE_NAME,
            SCHEMA_NAME,
            WAREHOUSE_NAME,
            EXECUTION_STATUS,
            TOTAL_ELAPSED_TIME / 1000 AS execution_time_seconds,
            BYTES_SCANNED,
            ROWS_PRODUCED,
            PARTITIONS_SCANNED,
            PARTITIONS_TOTAL,
            BYTES_SPILLED_TO_LOCAL_STORAGE,
            BYTES_SPILLED_TO_REMOTE_STORAGE,
            CREDITS_USED_CLOUD_SERVICES,
            START_TIME,
            END_TIME
        FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
        WHERE {where_clause}
        ORDER BY START_TIME DESC
        LIMIT {limit}
        """

        logger.info("Fetching query history (query_id=%s, fragment=%.50s...)", query_id, query_text_fragment)
        return self.execute_query(sql)

    def get_table_columns(self, database: str, schema: str, table_name: str) -> list[dict[str, Any]]:
        """
        Fetch column metadata for a specific table.

        Uses INFORMATION_SCHEMA for real-time results (no latency).

        Args:
            database: Database name.
            schema: Schema name.
            table_name: Table name.

        Returns:
            List of column metadata dicts.
        """
        sql = f"""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            ORDINAL_POSITION
        FROM {database}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{schema}'
          AND TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
        """
        return self.execute_query(sql)


def get_connection_manager() -> SnowflakeConnectionManager:
    """Return the process-wide singleton SnowflakeConnectionManager."""
    return SnowflakeConnectionManager()
