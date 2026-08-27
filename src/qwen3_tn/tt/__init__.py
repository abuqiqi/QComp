"""Public TT-matrix mathematics."""

from .decomposition import tt_svd_matrix
from .operations import (
    detensorize_matrix,
    reconstruct_matrix,
    slice_bond,
    tensorize_matrix,
    validate_cores,
)
from .spec import TTMatrixSpec

__all__ = [
    "TTMatrixSpec",
    "detensorize_matrix",
    "reconstruct_matrix",
    "slice_bond",
    "tensorize_matrix",
    "tt_svd_matrix",
    "validate_cores",
]
