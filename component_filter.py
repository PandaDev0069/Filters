"""Reusable algorithms from notebook cell 13; no notebook execution required."""

import math
import cv2
import numpy as np


def filter_components(mask, disk_area, cfg):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    minimum = max(
        int(cfg["min_component_area_px"]),
        math.ceil(disk_area * cfg["min_component_disk_fraction"]),
    )
    keep = stats[:, cv2.CC_STAT_AREA] >= minimum
    keep[0] = False
    result = keep[labels]
    return result, {
        "minimum_area_px": minimum,
        "raw_components": count - 1,
        "filtered_components": int(keep.sum()),
    }


def apply_component_filter(input_path, output_path, config=None, **options):
    """File API for the components stage; accepts mask_path/disk_path/reference_path."""
    from .cli import apply_stage
    return apply_stage("components", input_path, output_path, config=config, **options)


if __name__ == "__main__":
    from .cli import main
    main(["components", *__import__("sys").argv[1:]])
