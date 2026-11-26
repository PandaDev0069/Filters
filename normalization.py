"""Reusable algorithms from notebook cell 11; no notebook execution required."""

import numpy as np
import math


def robust_unit(image, mask, percentiles):
    image = np.asarray(image)
    mask = np.asarray(mask, dtype=bool)
    if image.ndim != 2 or mask.shape != image.shape:
        raise ValueError("Normalization requires an image and matching 2-D mask.")
    if (len(percentiles) != 2 or not np.isfinite(percentiles).all()
            or not 0 <= percentiles[0] < percentiles[1] <= 100):
        raise ValueError("Require 0 <= lower percentile < upper percentile <= 100.")
    values = np.asarray(image)[mask]
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Normalization requires finite on-disk pixels.")
    low, high = np.percentile(values, percentiles)
    result = np.zeros(image.shape, np.float32)
    if high - low <= 1e-6:
        result[mask] = 0.5
    else:
        result[mask] = np.clip((image[mask] - low) / (high - low), 0, 1)
    return result, {"low": float(low), "high": float(high)}


def to_uint8(unit):
    return np.rint(np.clip(unit, 0, 1) * 255).astype(np.uint8)


def normalized_enhanced_from_clahe(clahe, disk_mask, percentiles):
    if len(percentiles) != 2 or not all(math.isfinite(float(p)) for p in percentiles):
        raise ValueError("Provide two finite normalization percentiles.")
    if not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise ValueError("Require 0 <= lower percentile < upper percentile <= 100.")
    unit, stats = robust_unit(clahe, disk_mask, percentiles)
    enhanced = to_uint8(unit)
    if np.any(enhanced[~disk_mask]):
        raise AssertionError(
            "Normalized enhanced pixels outside the disk must remain zero."
        )
    return enhanced, stats


def apply_normalization(input_path, output_path, config=None, **options):
    """File API for the normalize stage; accepts mask_path/disk_path/reference_path."""
    from .cli import apply_stage
    return apply_stage("normalize", input_path, output_path, config=config, **options)


if __name__ == "__main__":
    from .cli import main
    main(["normalize", *__import__("sys").argv[1:]])
