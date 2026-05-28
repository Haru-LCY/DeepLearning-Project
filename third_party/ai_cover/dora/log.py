from __future__ import annotations


def fatal(message: str) -> None:
    raise SystemExit(message)


def bold(message: str) -> str:
    return message
