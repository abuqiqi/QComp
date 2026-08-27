"""Stable core API for TT-matrix compression."""

from .backends import available_tt_backends, create_tt_linear
from .model import TTTarget
from .tt import TTMatrixSpec, tt_svd_matrix

__all__ = [
    "TTMatrixSpec",
    "TTTarget",
    "available_tt_backends",
    "create_tt_linear",
    "tt_svd_matrix",
]
