"""Stable core API for TT-matrix compression."""

from .backends import (
    available_tt_backends,
    create_tt_linear,
    get_tt_backend,
    import_tt_backend_modules,
    probe_tt_backend,
    registered_tt_backends,
    register_tt_backend,
    tt_backend_metadata,
)
from .model import TTTarget
from .tt import TTMatrixSpec, tt_svd_matrix

__all__ = [
    "TTMatrixSpec",
    "TTTarget",
    "available_tt_backends",
    "create_tt_linear",
    "get_tt_backend",
    "import_tt_backend_modules",
    "probe_tt_backend",
    "registered_tt_backends",
    "register_tt_backend",
    "tt_backend_metadata",
    "tt_svd_matrix",
]
