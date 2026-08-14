"""Stable, safe errors exposed at the command-line boundary."""

from __future__ import annotations


class CliError(Exception):
    """An expected CLI failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str, exit_code: int = 3) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
