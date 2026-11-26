"""Reusable algorithms from notebook cell 11; no notebook execution required."""

import cv2
import numpy as np


def radial_correction(image, rho, disk_mask, cfg):
    stride = cfg["radial_sample_stride"]
    radii = rho[::stride, ::stride]
    intensities = image[::stride, ::stride].astype(np.float32)
    bins = cfg["radial_bins"]
    centers = (np.arange(bins) + 0.5) / bins
    profile = np.full(bins, np.nan, np.float32)
    for i in range(bins):
        pixels = intensities[(radii >= i / bins) & (radii < (i + 1) / bins)]
        if pixels.size >= 4:
            profile[i] = np.median(pixels)
    good = np.isfinite(profile) & (profile > 1)
    if good.sum() < max(4, bins // 4):
        raise ValueError(
            "Insufficient radial samples; adjust radial sampling settings."
        )
    profile = np.interp(centers, centers[good], profile[good]).astype(np.float32)
    profile = cv2.GaussianBlur(profile[None, :], (9, 1), 0).ravel()
    target = float(np.median(profile[centers < 0.5]))
    background = (
        np.interp(rho.ravel(), centers, profile).reshape(image.shape).astype(np.float32)
    )
    gain = np.clip(target / np.maximum(background, 1), *cfg["radial_gain_limits"])
    corrected = image.astype(np.float32) * gain
    corrected[~disk_mask] = 0
    return corrected, centers, profile


def apply_radial_correction(input_path, output_path, config=None, **options):
    """File API for the radial stage; accepts mask_path/disk_path/reference_path."""
    from .cli import apply_stage
    return apply_stage("radial", input_path, output_path, config=config, **options)


if __name__ == "__main__":
    from .cli import main
    main(["radial", *__import__("sys").argv[1:]])
