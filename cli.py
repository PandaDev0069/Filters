"""File-based access to individual stages; array APIs remain in each module."""

import argparse
from pathlib import Path
import sys
import numpy as np
from .config import get_config, RADIUS_EDGE_CONFIG
from .image_io import (load_array, load_grayscale, load_mask, read_json,
                       save_array, write_json)

STAGES = ("disk", "median", "radial", "polynomial", "prepare-clahe", "normalize", "clahe",
          "bilateral", "laplacian", "threshold", "components", "closing",
          "size", "radius", "instances", "pipeline")


def apply_stage(stage, input_path, output_path, *, config=None, preset="baseline",
                mask_path=None, disk_path=None, reference_path=None, rule_path=None):
    """Apply only the named stage and save its output plus a JSON sidecar.

    Supply the preceding stage's output as input; this function does not silently
    rerun upstream corrections. Floating stages and labels require .npy/.npz.
    Disk geometry can come from the disk stage's JSON or an original reference.
    """
    from .solar_disk_filter import detect_solar_disk

    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; choose from {STAGES}.")
    cfg = get_config(preset, **(config or {}))
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Choose an output path different from the input.")
    binary_stages = {"components", "closing", "size", "radius", "instances"}
    if stage in binary_stages:
        image = load_mask(input_path)
    elif input_path.suffix.lower() in {".npy", ".npz"}:
        image = load_array(input_path)
    else:
        image = load_grayscale(input_path, cfg)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Stages require a finite two-dimensional array.")
    mask = load_mask(mask_path, image.shape) if mask_path else np.ones(image.shape, bool)
    metadata = {}

    def geometry():
        if disk_path:
            data = read_json(disk_path)
            disk = data.get("disk", data.get("metadata", {}).get("disk", data))
            cx, cy, radius = (float(disk[key]) for key in ("cx", "cy", "radius"))
            if not np.isfinite([cx, cy, radius]).all() or radius <= 0:
                raise ValueError("Disk geometry must have finite center and positive radius.")
            yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
            rho = (np.hypot(xx - cx, yy - cy) / radius).astype(np.float32)
            return disk, rho, mask if mask_path else rho <= 1
        if reference_path:
            reference = load_grayscale(reference_path, cfg)
        elif image.dtype == np.uint8:
            reference = image
        else:
            raise ValueError("Supply --disk or an original --reference image for disk geometry.")
        if reference.shape != image.shape:
            raise ValueError("Reference must have the same shape as the stage input.")
        disk, rho, detected_mask = detect_solar_disk(reference, cfg)
        return disk, rho, mask if mask_path else detected_mask

    if stage == "pipeline":
        from .pipeline import preprocess_image
        from .export import save_image_result
        result = preprocess_image(image, cfg)
        save_image_result(result, output_path, cfg, source_path=input_path)
        return result
    if stage == "disk":
        disk, _, output = geometry()
        metadata["disk"] = disk
    elif stage == "median":
        from .median_filter import median_filter
        output = median_filter(image, cfg["median_kernel"], mask if mask_path else None)
    elif stage == "radial":
        from .radial_correction import radial_correction
        disk, rho, disk_mask = geometry()
        output, centers, profile = radial_correction(image, rho, disk_mask, cfg)
        metadata.update(disk=disk, radial_centers=centers.tolist(), radial_profile=profile.tolist())
    elif stage == "polynomial":
        from .polynomial_correction import polynomial_correction
        disk, _, disk_mask = geometry()
        output, background, coefficients = polynomial_correction(image, disk_mask, disk, cfg)
        metadata.update(disk=disk, polynomial_coefficients=coefficients,
                        background_file=output_path.stem + ".background.npy")
        save_array(output_path.with_name(metadata["background_file"]), background)
    elif stage == "prepare-clahe":
        from .clahe_filter import prepare_fixed_clahe_input
        output, metadata = prepare_fixed_clahe_input(image, mask, cfg["normalization_percentiles"])
    elif stage == "normalize":
        from .normalization import robust_unit, to_uint8
        output, metadata = robust_unit(
            image, mask, cfg.get("post_clahe_normalization_percentiles", cfg["normalization_percentiles"]))
        output = to_uint8(output)
    elif stage == "clahe":
        from .clahe_filter import clahe_filter
        output = clahe_filter(image, cfg["clahe_clip_limit"], cfg["clahe_grid"], mask)
    elif stage == "bilateral":
        from .bilateral_filter import bilateral_filter
        setting = cfg.get("bilateral_filter") or {}
        output = bilateral_filter(image, setting.get("diameter", 5), setting.get("sigma_color", 15),
                                  setting.get("sigma_space", 3), mask if mask_path else None)
    elif stage == "laplacian":
        from .laplacian_filter import signed_laplacian, LAPLACIAN_KERNEL, effective_signed_kernel
        spec = cfg.get("signed_laplacian_variant") or {}
        kernel = effective_signed_kernel({"kernel": spec.get("raw_kernel", spec.get("kernel", LAPLACIAN_KERNEL)),
                                          "gain": spec.get("gain", 1)})
        output = signed_laplacian(image, kernel)
        output[~mask] = 0
        metadata["kernel"] = kernel.tolist()
    elif stage == "threshold":
        from .adaptive_threshold import adaptive_candidates
        spec = cfg.get("signed_laplacian_variant")
        kernel = None
        if spec:
            from .laplacian_filter import effective_signed_kernel, LAPLACIAN_KERNEL
            kernel = effective_signed_kernel({"kernel": spec.get("raw_kernel", spec.get("kernel", LAPLACIAN_KERNEL)),
                                              "gain": spec.get("gain", 1)})
        _, _, output, metadata = adaptive_candidates(image, mask, cfg, kernel)
    elif stage == "components":
        from .component_filter import filter_components
        output, metadata = filter_components(image, int(mask.sum()), cfg)
    elif stage == "closing":
        from .morphological_closing import conservative_closing
        from .adaptive_threshold import candidate_region
        output = conservative_closing(image, candidate_region(mask, cfg["candidate_limb_margin_px"]), cfg)
    elif stage == "size":
        from .annotation_size_filter import filter_candidate_size_bounds
        rule = read_json(rule_path) if rule_path else cfg.get("annotation_area_filter")
        if not rule:
            raise ValueError("Supply --rule fitted-size-bounds.json or --preset recommended.")
        output, metadata = filter_candidate_size_bounds(
            image, rule["minimum_candidate_area_px"], rule["maximum_candidate_area_px"],
            rule["maximum_candidate_bbox_span_px"])
    elif stage == "radius":
        from .radius_edge_filter import radius_edge_geometry, reject_radius_edge_components
        disk, _, _ = geometry()
        safe, _, info = radius_edge_geometry(image.shape, disk, cfg.get("radius_edge_filter") or RADIUS_EDGE_CONFIG)
        output, _, metadata = reject_radius_edge_components(image, safe)
        metadata.update(info)
    elif stage == "instances":
        from .instance_segmentation import candidates_to_instances
        output, stats, count = candidates_to_instances(image)
        metadata = {"instances": count, "component_stats_xywh_area": stats.tolist()}
    save_array(output_path, output)
    write_json(output_path.with_suffix(output_path.suffix + ".json"), {
        "stage": stage, "source": str(input_path.resolve()), "config": cfg, "metadata": metadata})
    return output, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("input", help="Input image, binary PNG, or exact .npy/.npz stage")
    parser.add_argument("output", help="Output .png/.npy/.npz, or directory for pipeline")
    parser.add_argument("--preset", choices=("baseline", "recommended"), default="baseline")
    parser.add_argument("--config", help="JSON config object, or a notebook export with parameters")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                        help="Override a config setting; repeat for multiple settings")
    parser.add_argument("--mask", help="Binary disk mask")
    parser.add_argument("--disk", help="Disk geometry JSON, including disk-stage sidecars")
    parser.add_argument("--reference", help="Original grayscale image used to detect the disk")
    parser.add_argument("--rule", help="Training-fitted size rule JSON")
    args = parser.parse_args(argv)
    import json
    try:
        config = read_json(args.config) if args.config else {}
        config = config.get("parameters", config)
        for assignment in args.set:
            key, sep, value = assignment.partition("=")
            if not sep:
                raise ValueError("Overrides use KEY=JSON, for example median_kernel=5.")
            config[key] = json.loads(value)
        apply_stage(args.stage, args.input, args.output, config=config, preset=args.preset,
                    mask_path=args.mask, disk_path=args.disk, reference_path=args.reference,
                    rule_path=args.rule)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Saved {args.stage}: {args.output}")


if __name__ == "__main__":
    main()
