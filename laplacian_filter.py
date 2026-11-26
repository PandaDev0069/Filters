"""Sections 5/14: dark-positive signed Laplacians and adjustable kernels."""

import math
import cv2
import numpy as np
from .normalization import to_uint8

LAPLACIAN_KERNEL = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)


def make_neighbor_laplacian(size=3, diagonal_weight=1.0):
    """Build the notebook's 3x3 cross/weighted or wider surround stencil."""
    if int(size) != size or size < 3 or size % 2 != 1:
        raise ValueError("Signed Laplacian support must be odd and at least 3.")
    if not math.isfinite(diagonal_weight) or diagonal_weight < 0:
        raise ValueError("Diagonal weights must be finite and nonnegative.")
    kernel = np.ones((int(size), int(size)), np.float32)
    center = int(size) // 2
    if size == 3:
        kernel[0, 0] = kernel[0, 2] = kernel[2, 0] = kernel[2, 2] = diagonal_weight
    kernel[center, center] = 0
    kernel[center, center] = -float(kernel.sum(dtype=np.float64))
    return kernel


def effective_signed_kernel(variant):
    """Normalize positive neighbor mass to 8, then apply a positive gain."""
    kernel = np.asarray(variant["kernel"], np.float32)
    gain = float(variant.get("gain", 1))
    if (kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]
            or kernel.shape[0] < 3 or kernel.shape[0] % 2 != 1
            or not np.isfinite(kernel).all()):
        raise ValueError("Provide a finite, odd square kernel of support >= 3.")
    center = kernel.shape[0] // 2
    neighbors = kernel.copy()
    neighbors[center, center] = 0
    mass = float(neighbors.sum(dtype=np.float64))
    if (kernel[center, center] >= 0 or np.any(neighbors < 0) or mass <= 0
            or not np.isclose(kernel.sum(dtype=np.float64), 0, atol=1e-5)
            or not math.isfinite(gain) or gain <= 0):
        raise ValueError("Use a zero-sum dark-positive stencil and a positive finite gain.")
    return (kernel * (8.0 / mass) * gain).astype(np.float32)


def signed_laplacian(enhanced, kernel=None):
    """Filter uint8 grayscale on the [0, 1] scale; return signed float32."""
    if enhanced.ndim != 2 or enhanced.dtype != np.uint8:
        raise ValueError("Signed Laplacian expects a two-dimensional uint8 image.")
    kernel = LAPLACIAN_KERNEL if kernel is None else np.asarray(kernel, np.float32)
    return cv2.filter2D(enhanced.astype(np.float32) / 255.0, cv2.CV_32F,
                        kernel, borderType=cv2.BORDER_REFLECT_101)


def signed_laplacian_view(laplacian, disk_mask):
    """Centered uint8 display only; retain the original signed array for analysis."""
    limit = max(float(np.percentile(np.abs(laplacian[disk_mask]), 99)), 1e-6)
    view = to_uint8(0.5 + 0.5 * laplacian / limit)
    view[~disk_mask] = 0
    return view


def apply_laplacian_filter(input_path, output_path, config=None, **options):
    """Save a signed .npy/.npz response from an enhanced image."""
    from .cli import apply_stage
    return apply_stage("laplacian", input_path, output_path, config=config, **options)


if __name__ == "__main__":
    from .cli import main
    main(["laplacian", *__import__("sys").argv[1:]])
