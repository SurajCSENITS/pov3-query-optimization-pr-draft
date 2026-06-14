#!/usr/bin/env python3
"""
Snowflake Connectivity Test Script.

Run this after configuring your .env file to verify that:
1. Credentials are valid
2. The connection can be established
3. The required ACCOUNT_USAGE views are accessible

Usage:
    python scripts/test_snowflake_connection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.config.settings import get_settings
from src.connectors.snowflake_manager import get_connection_manager

console = Console()


def main() -> None:
    console.print(
        Panel(
            "[bold cyan]Snowflake Connectivity Test[/]",
            border_style="cyan",
            expand=False,
        )
    )

    # ── Step 1: Check configuration ─────────────────────────────
    settings = get_settings()

    config_table = Table(
        title="Configuration",
        box=box.ROUNDED,
        border_style="blue",
    )
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    config_table.add_row("Account", settings.snowflake_account or "[red]NOT SET[/]")
    config_table.add_row("User", settings.snowflake_user or "[red]NOT SET[/]")
    config_table.add_row("Password", "***" if settings.snowflake_password else "[red]NOT SET[/]")
    config_table.add_row("Warehouse", settings.snowflake_warehouse)
    config_table.add_row("Database", settings.snowflake_database)
    config_table.add_row("Schema", settings.snowflake_schema)
    config_table.add_row("Role", settings.snowflake_role)
    config_table.add_row("Enabled", str(settings.snowflake_enabled))
    config_table.add_row("Configured", str(settings.snowflake_configured))
    console.print(config_table)
    console.print()

    if not settings.snowflake_configured:
        console.print(
            "[yellow]⚠️  Snowflake is not configured or not enabled.[/]\n"
            "Please ensure your .env file has valid credentials and "
            "SNOWFLAKE_ENABLED=true"
        )
        return

    # ── Step 2: Test basic connection ───────────────────────────
    console.print("[bold]Test 1: Basic Connection...[/]")
    manager = get_connection_manager()

    if manager.health_check():
        console.print("  ✅ Connection established successfully\n")
    else:
        console.print("  ❌ Connection failed\n")
        return

    # ── Step 3: Test Snowflake version ──────────────────────────
    console.print("[bold]Test 2: Snowflake Version...[/]")
    try:
        result = manager.execute_query(
            "SELECT CURRENT_VERSION() AS version, "
            "CURRENT_ACCOUNT() AS account, "
            "CURRENT_WAREHOUSE() AS warehouse, "
            "CURRENT_ROLE() AS role"
        )
        if result:
            row = result[0]
            console.print(f"  Version:   {row.get('VERSION')}")
            console.print(f"  Account:   {row.get('ACCOUNT')}")
            console.print(f"  Warehouse: {row.get('WAREHOUSE')}")
            console.print(f"  Role:      {row.get('ROLE')}")
            console.print("  ✅ Passed\n")
    except Exception as e:
        console.print(f"  ❌ Failed: {e}\n")

    # ── Step 4: Test ACCOUNT_USAGE access ───────────────────────
    console.print("[bold]Test 3: ACCOUNT_USAGE Access...[/]")
    try:
        result = manager.execute_query(
            "SELECT COUNT(*) AS cnt "
            "FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY "
            "WHERE START_TIME >= DATEADD(hour, -24, CURRENT_TIMESTAMP()) "
            "LIMIT 1"
        )
        count = result[0]["CNT"] if result else 0
        console.print(f"  Queries in last 24h: {count}")
        console.print("  ✅ ACCOUNT_USAGE access confirmed\n")
    except Exception as e:
        console.print(f"  ❌ ACCOUNT_USAGE access denied: {e}")
        console.print(
            "  [yellow]Run this SQL as ACCOUNTADMIN:[/]\n"
            "  GRANT IMPORTED PRIVILEGES ON DATABASE SNOWFLAKE "
            "TO ROLE POV3_AGENT_ROLE;\n"
        )

    # ── Step 5: Test EXPLAIN capability ─────────────────────────
    console.print("[bold]Test 4: EXPLAIN Capability...[/]")
    try:
        result = manager.execute_query("EXPLAIN USING TEXT SELECT 1")
        if result:
            console.print(f"  Plan rows returned: {len(result)}")
            console.print("  ✅ EXPLAIN works\n")
    except Exception as e:
        console.print(f"  ❌ EXPLAIN failed: {e}\n")

    # ── Cleanup ─────────────────────────────────────────────────
    manager.close()
    console.print("[bold green]All tests complete.[/]")


if __name__ == "__main__":
    main()
