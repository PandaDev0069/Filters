"""Reusable algorithms from notebook cell 38; no notebook execution required."""

import copy
import math
import time
from pathlib import Path
import cv2
import numpy as np


def radius_edge_geometry(shape, disk, cfg):
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Radius filtering requires a nonempty 2-D image shape.")
    cx, cy, radius = (float(disk[key]) for key in ("cx", "cy", "radius"))
    minimum_width = float(cfg["minimum_edge_width_px"])
    fraction = float(cfg["edge_width_radius_fraction"])
    if not all(math.isfinite(v) for v in (cx, cy, radius, minimum_width, fraction)):
        raise ValueError("Disk geometry and edge-width settings must be finite.")
    if radius <= 0 or minimum_width <= 0 or not 0 <= fraction < 1:
        raise ValueError("Require positive radius/edge width and a radius fraction in [0, 1).")
    if not 0 <= cx < shape[1] or not 0 <= cy < shape[0]:
        raise ValueError("Detected solar center must lie inside the image.")
    if cfg["connectivity"] != 8 or cfg["removal_mode"] != "whole_component_touching_or_outside_safe_radius":
        raise ValueError("This experiment requires whole-component removal with 8-connectivity.")
    width = max(minimum_width, radius * fraction)
    safe_radius = radius - width
    if safe_radius <= 0:
        raise ValueError("The edge band must leave a positive safe interior radius.")
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    squared_distance = (xx - cx) ** 2 + (yy - cy) ** 2
    safe_interior = squared_distance < safe_radius ** 2
    edge_band = (squared_distance >= safe_radius ** 2) & (squared_distance <= radius ** 2)
    geometry = {
        "cx": cx, "cy": cy, "disk_radius_px": radius,
        "edge_width_px": width, "safe_radius_px": safe_radius,
        "safe_interior_pixels": int(safe_interior.sum()),
        "edge_band_pixels": int(edge_band.sum()),
    }
    return safe_interior, edge_band, geometry


def reject_radius_edge_components(mask, safe_interior):
    mask, safe_interior = np.asarray(mask, bool), np.asarray(safe_interior, bool)
    if mask.ndim != 2 or mask.shape != safe_interior.shape:
        raise ValueError("Candidate mask and safe interior must have the same 2-D shape.")
    count, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    touched = np.unique(labels[mask & ~safe_interior])
    keep = np.ones(count, bool)
    keep[0] = False
    keep[touched] = False # type: ignore
    retained = keep[labels] # type: ignore
    removed = mask & ~retained
    if np.any(retained & ~safe_interior):
        raise AssertionError("Radius-edge rejection left candidates outside the safe interior.")
    return retained, removed, {
        "components_before": count - 1,
        "components_after": int(keep.sum()),
        "edge_components_removed": int(len(touched)),
        "pixels_removed": int(removed.sum()),
        "edge_or_outside_pixels_removed": int(np.count_nonzero(removed & ~safe_interior)),
        "interior_pixels_removed_with_edge_components": int(np.count_nonzero(removed & safe_interior)),
        "remaining_edge_or_outside_pixels": int(np.count_nonzero(retained & ~safe_interior)),
    }


def apply_radius_edge_filter(result, cfg):
    started = time.perf_counter()
    safe_interior, edge_band, geometry = radius_edge_geometry(
        result["candidates"].shape, result["metadata"]["disk"], cfg
    )
    filtered_result = dict(result)
    filtered_result["filtered_candidates"], _, intermediate_counts = reject_radius_edge_components(
        result["filtered_candidates"], safe_interior
    )
    filtered_result["candidates"], removed, counts = reject_radius_edge_components(
        result["candidates"], safe_interior
    )
    filtered_result["edge_safe_disk_mask"] = safe_interior
    filtered_result["edge_band_mask"] = edge_band
    filtered_result["edge_removed_candidates"] = removed
    filtered_result["multi_channel"] = np.stack(
        [result["original"], result["enhanced"], filtered_result["candidates"].astype(np.uint8) * 255], axis=-1
    )
    elapsed = time.perf_counter() - started
    diagnostics = {**geometry, **counts, "filter_seconds": elapsed}
    filtered_result["metadata"] = copy.deepcopy(result["metadata"])
    filtered_result["metadata"]["radius_edge_filter"] = {**cfg, **diagnostics}
    filtered_result["metadata"]["components"]["filtered_components"] = intermediate_counts["components_after"]
    filtered_result["metadata"]["preprocessing_seconds"] += elapsed
    return filtered_result, diagnostics


def apply_radius_edge_filter_file(input_path, output_path, config=None, **options):
    """File API for the radius stage; accepts mask_path/disk_path/reference_path."""
    from .cli import apply_stage
    return apply_stage("radius", input_path, output_path, config=config, **options)


if __name__ == "__main__":
    from .cli import main
    main(["radius", *__import__("sys").argv[1:]])
