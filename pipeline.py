"""Composable image-local pipeline for baseline and selected notebook settings.

Individual algorithms live in their own modules. No notebook, cache, dataset,
annotations, or plots are loaded when this module is imported.
"""

import copy
import time
import numpy as np
from .config import get_config
from .solar_disk_filter import detect_solar_disk
from .median_filter import median_filter
from .bilateral_filter import bilateral_filter
from .radial_correction import radial_correction
from .polynomial_correction import polynomial_correction
from .normalization import normalized_enhanced_from_clahe
from .clahe_filter import prepare_fixed_clahe_input, clahe_filter
from .laplacian_filter import LAPLACIAN_KERNEL, effective_signed_kernel
from .adaptive_threshold import adaptive_candidates
from .component_filter import filter_components
from .morphological_closing import conservative_closing
from .annotation_size_filter import filter_candidate_size_bounds
from .radius_edge_filter import apply_radius_edge_filter


def preprocess_image(image, cfg=None, *, preset="baseline", geometry=None):
    """Return all stages and diagnostics, preserving the original pixel grid.

    geometry is an optional (disk, rho, disk_mask) tuple detected on the original
    image. Sweeps use it to freeze geometry across variants. Bilateral is applied
    once at the configured position; baseline and recommended presets disable it.
    """
    cfg = get_config(preset, **(cfg or {}))
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("preprocess_image expects a 2-D uint8 grayscale array.")
    if cfg["strict_shape"] and tuple(image.shape) != tuple(cfg["expected_shape"]):
        raise ValueError(f"Expected image shape {cfg['expected_shape']}, got {image.shape}.")
    started = time.perf_counter()
    disk, rho, mask = detect_solar_disk(image, cfg) if geometry is None else geometry
    if rho.shape != image.shape or mask.shape != image.shape or mask.dtype != bool:
        raise ValueError("Frozen disk geometry must match the source image.")
    bilateral = cfg.get("bilateral_filter") or {}
    position = bilateral.get("position", "after_normalization") if bilateral.get("enabled", False) else "none"
    if position not in {"none", "before_median", "after_median", "before_clahe", "after_normalization"}:
        raise ValueError("Unknown bilateral insertion position.")

    def smooth(array, preserve_exterior=False):
        return bilateral_filter(array, bilateral.get("diameter", 5),
                                bilateral.get("sigma_color", 15), bilateral.get("sigma_space", 3),
                                mask, preserve_exterior)

    result = {"original": image.copy(), "disk_mask": mask.copy(), "rho": rho.copy(),
              "masked": np.where(mask, image, 0).astype(np.uint8)}
    median_input = image
    if position == "before_median":
        median_input = smooth(image, preserve_exterior=True)
        result.update(bilateral_input=image.copy(), bilateral=median_input)
    median = median_filter(median_input, cfg["median_kernel"], mask)
    radial_input = median
    if position == "after_median":
        radial_input = smooth(median)
        result.update(bilateral_input=median, bilateral=radial_input)
    radial, centers, profile = radial_correction(radial_input, rho, mask, cfg)
    corrected, background, coefficients = polynomial_correction(radial, mask, disk, cfg)
    clahe_input, before_stats = prepare_fixed_clahe_input(corrected, mask, cfg["normalization_percentiles"])
    if position == "before_clahe":
        previous = clahe_input
        clahe_input = smooth(previous, preserve_exterior=True)
        result.update(bilateral_input=previous, bilateral=clahe_input)
    clahe = clahe_filter(clahe_input, cfg["clahe_clip_limit"], cfg["clahe_grid"], mask)
    enhanced, after_stats = normalized_enhanced_from_clahe(
        clahe, mask, cfg.get("post_clahe_normalization_percentiles", cfg["normalization_percentiles"]))
    laplacian_input = enhanced
    if position == "after_normalization":
        laplacian_input = smooth(enhanced)
        result.update(bilateral_input=enhanced, bilateral=laplacian_input)
    spec = cfg.get("signed_laplacian_variant")
    kernel = LAPLACIAN_KERNEL
    if spec:
        kernel = effective_signed_kernel({"kernel": spec.get("raw_kernel", spec.get("kernel", LAPLACIAN_KERNEL)),
                                          "gain": spec.get("gain", 1)})
    laplacian, eligible, raw, thresholds = adaptive_candidates(laplacian_input, mask, cfg, kernel)
    filtered, components = filter_components(raw, int(mask.sum()), cfg)
    candidates = conservative_closing(filtered, eligible, cfg)
    metadata = {
        "disk": copy.deepcopy(disk), "shape": list(image.shape),
        "radial_centers": centers.tolist(), "radial_profile": profile.tolist(),
        "polynomial_coefficients": coefficients,
        "normalization": {"pre_clahe": before_stats, "post_clahe": after_stats},
        "threshold_selection": thresholds, "components": components,
        "bilateral_location": {**copy.deepcopy(bilateral), "position": position, "enabled": position != "none"},
        "laplacian_kernel": kernel.tolist(), "preprocessing_seconds": 0.0,
    }
    rule = cfg.get("annotation_area_filter")
    if rule:
        args = (rule["minimum_candidate_area_px"], rule["maximum_candidate_area_px"],
                rule["maximum_candidate_bbox_span_px"])
        filtered, filtered_counts = filter_candidate_size_bounds(filtered, *args)
        candidates, counts = filter_candidate_size_bounds(candidates, *args)
        metadata["annotation_area_filter"] = {**copy.deepcopy(rule), **counts}
        metadata["components"].update(
            minimum_area_px=max(components["minimum_area_px"], int(args[0])),
            maximum_area_px=int(args[1]), maximum_bbox_span_px=int(args[2]),
            filtered_components=filtered_counts["components_after"],
        )
    result.update(median=median, radial_input=radial_input, radial=radial,
                  corrected=corrected, background=background, clahe_input=clahe_input,
                  clahe=clahe, enhanced=enhanced, laplacian_input=laplacian_input,
                  laplacian=laplacian, eligible=eligible, raw_candidates=raw,
                  filtered_candidates=filtered, candidates=candidates, metadata=metadata)
    if cfg.get("radius_edge_filter"):
        result, _ = apply_radius_edge_filter(result, cfg["radius_edge_filter"])
    result["multi_channel"] = np.stack(
        [image, laplacian_input, result["candidates"].astype(np.uint8) * 255], axis=-1)
    result["metadata"]["preprocessing_seconds"] = time.perf_counter() - started
    result["metadata"]["channel_order_rgb"] = ["original", "laplacian_input", "candidates"]
    return result


if __name__ == "__main__":
    from .cli import main
    main(["pipeline", *__import__("sys").argv[1:]])
