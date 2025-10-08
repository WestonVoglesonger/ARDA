"""
Compatibility shim exposing the existing `alg2sv` package under the future
`arda` namespace.

This allows early adopters to import `arda` while the internal modules remain
in the legacy package until the full rename is complete.
"""

from alg2sv import *  # noqa: F401,F403

__all__ = [name for name in dir() if not name.startswith('_')]
