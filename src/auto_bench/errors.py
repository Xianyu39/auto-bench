from __future__ import annotations


class AutobenchError(Exception):
    """Base error for user-facing autobench failures."""


class ProtocolError(AutobenchError):
    """Raised when an experiment YAML file violates the protocol."""
