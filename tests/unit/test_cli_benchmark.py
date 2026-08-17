"""Regression guard for offline benchmark timestamp construction."""

from __future__ import annotations

from legal_rag.cli import _operational_timestamp


def test_operational_timestamp_needs_no_external_timezone_database() -> None:
    assert _operational_timestamp().endswith("+07:00")
