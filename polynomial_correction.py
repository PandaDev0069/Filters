"""Reusable algorithms from notebook cell 11; no notebook execution required."""

import numpy as np


def polynomial_correction(image, disk_mask, disk, cfg):
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    x = ((xx - disk["cx"]) / disk["radius"]).astype(np.float32)
    y = ((yy - disk["cy"]) / disk["radius"]).astype(np.float32)
    stride = cfg["polynomial_sample_stride"]
    sampled_mask = disk_mask[::stride, ::stride]
    xs = np.broadcast_to(x[:, ::stride], sampled_mask.shape)[sampled_mask]
    ys = np.broadcast_to(y[::stride, :], sampled_mask.shape)[sampled_mask]
    values = image[::stride, ::stride][sampled_mask].astype(np.float64)
    degree = cfg["polynomial_degree"]
    powers = [
        (i, j)
        for total in range(degree + 1)
        for i in range(total + 1)
        for j in [total - i]
    ]
    design = np.column_stack([xs**i * ys**j for i, j in powers]).astype(np.float64)
    low, high = np.percentile(values, [10, 90])
    keep = (values >= low) & (values <= high)
    coefficients = None
    for _ in range(cfg["polynomial_iterations"]):
        if keep.sum() < 3 * len(powers):
            raise ValueError("Insufficient pixels for polynomial background fit.")
        coefficients, _, rank, _ = np.linalg.lstsq(
            design[keep], values[keep], rcond=None
        )
        if rank != len(powers):
            raise ValueError("Polynomial background fit is rank deficient.")
        residual = values - design @ coefficients
        center = np.median(residual[keep])
        mad = max(1.4826 * np.median(np.abs(residual[keep] - center)), 1e-3)
        keep = np.abs(residual - center) <= cfg["polynomial_clip_sigma"] * mad
    background = np.zeros(image.shape, np.float32)
    for coefficient, (i, j) in zip(coefficients, powers):
        background += np.float32(coefficient) * (x**i) * (y**j)
    target = float(np.median(background[disk_mask]))
    corrected = image - background + target
    corrected[~disk_mask] = 0
    background[~disk_mask] = 0
    return corrected.astype(np.float32), background, coefficients.tolist()


def apply_polynomial_correction(input_path, output_path, config=None, **options):
    """File API for the polynomial stage; accepts mask_path/disk_path/reference_path."""
    from .cli import apply_stage
    return apply_stage("polynomial", input_path, output_path, config=config, **options)


if __name__ == "__main__":
    from .cli import main
    main(["polynomial", *__import__("sys").argv[1:]])
