"""Reusable algorithms from notebook cell 13; no notebook execution required."""

import cv2
import numpy as np


def conservative_closing(mask, eligible, cfg):
    size = int(cfg["closing_kernel"])
    if size == 0:
        return mask.copy()
    if size != 3:
        raise ValueError("Conservative closing supports 0 (off) or 3 (one iteration).")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return (
        cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1
        ).astype(bool)
        & eligible
    )


def apply_morphological_closing(input_path, output_path, config=None, **options):
    """File API for the closing stage; accepts mask_path/disk_path/reference_path."""
    from .cli import apply_stage
    return apply_stage("closing", input_path, output_path, config=config, **options)


if __name__ == "__main__":
    from .cli import main
    main(["closing", *__import__("sys").argv[1:]])
