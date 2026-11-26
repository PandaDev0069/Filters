"""Sections 9/14–18: independent parameter trials using the reusable pipeline."""

import copy
import csv
from itertools import product
from pathlib import Path
import re
import numpy as np
from .config import get_config
from .image_io import load_grayscale, load_mask, write_json
from .pipeline import preprocess_image
from .solar_disk_filter import detect_solar_disk


def laplacian_variants():
    from .laplacian_filter import make_neighbor_laplacian
    settings = [("baseline_8n", 3, 1, 1), ("cross_4n", 3, 0, 1),
                ("diagonal_025", 3, 0.25, 1), ("diagonal_2", 3, 2, 1),
                ("neighbors_5x5", 5, 1, 1), ("neighbors_7x7", 7, 1, 1),
                ("baseline_gain_05", 3, 1, 0.5), ("baseline_gain_2", 3, 1, 2)]
    return [{"id": name, "overrides": {"signed_laplacian_variant": {
        "raw_kernel": make_neighbor_laplacian(size, diagonal).tolist(), "gain": gain}}}
        for name, size, diagonal, gain in settings]


def normalization_variants(reference=(1, 99.5)):
    ranges = [list(reference), [0, 100], [0.5, 99.5], [2, 99.5], [5, 99.5],
              [1, 95], [1, 99], [1, 99.9], [2, 98]]
    unique = list(dict.fromkeys(tuple(pair) for pair in ranges))
    return [{"id": f"p{low:g}_{high:g}", "overrides": {
        "post_clahe_normalization_percentiles": [low, high]}} for low, high in unique]


def clahe_variants(reference_clip=2, reference_grid=(8, 8)):
    settings = [(reference_clip, tuple(reference_grid))]
    settings += [(clip, (grid, grid)) for clip, grid in product((1, 2, 4), (4, 8, 16))]
    return [{"id": f"clip{clip:g}_grid{grid[0]}x{grid[1]}",
             "overrides": {"clahe_clip_limit": clip, "clahe_grid": list(grid)}}
            for clip, grid in dict.fromkeys(settings)]


def bilateral_variants():
    variants = [{"id": "reference", "overrides": {"bilateral_filter": {"enabled": False}}}]
    for diameter, color, space in product((5, 9), (15, 35), (3, 7)):
        variants.append({"id": f"d{diameter}_c{color}_s{space}", "overrides": {
            "bilateral_filter": {"enabled": True, "position": "after_normalization",
                                 "diameter": diameter, "sigma_color": color, "sigma_space": space}}})
    return variants


def bilateral_location_variants(diameter=5, sigma_color=15, sigma_space=3):
    return [{"id": position, "overrides": {"bilateral_filter": {
        "enabled": position != "none", "position": position, "diameter": diameter,
        "sigma_color": sigma_color, "sigma_space": sigma_space}}}
        for position in ("none", "after_normalization", "after_median", "before_clahe", "before_median")]


def run_sweep(images, variants, *, config=None, preset="baseline", truths=None,
              output_dir=None, freeze_geometry=True):
    """Run independent variants on a name -> array/path mapping.

    Each trial starts from the original and a fresh config. Optional truths map
    the same names to full-grid binary arrays/paths. No label data enters filters.
    Scores describe the supplied images; this function does not assign data splits.
    A truth may be one annotator's union; run annotators separately to preserve scope.
    """
    cfg = get_config(preset, **(config or {}))
    variants = copy.deepcopy(list(variants))
    if not variants or len({v["id"] for v in variants}) != len(variants):
        raise ValueError("Provide at least one variant with unique IDs.")
    disk_keys = {"hough_coarse_size", "hough_fine_size", "hough_param1", "hough_param2", "disk_radius_fraction"}
    for variant in variants:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", variant["id"]) or variant["id"] in {".", ".."}:
            raise ValueError("Variant IDs must be simple filename-safe identifiers.")
        get_config(**{**cfg, **variant["overrides"]})
        if freeze_geometry and disk_keys.intersection(variant["overrides"]):
            raise ValueError("Set freeze_geometry=False when tuning disk-detection parameters.")
    root = Path(output_dir) if output_dir is not None else None
    if root:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "experiment_config.json", {
            "config": cfg, "variants": variants, "freeze_geometry": freeze_geometry,
            "scored": truths is not None, "images": list(images),
        })
    rows = []
    output_names = set()
    for name, source in images.items():
        stem = Path(str(name)).stem
        if stem in {"", ".", ".."} or stem in output_names:
            raise ValueError("Image names must have distinct nonempty filename stems.")
        output_names.add(stem)
        image = source if isinstance(source, np.ndarray) else load_grayscale(source, cfg)
        geometry = detect_solar_disk(image, cfg) if freeze_geometry else None
        truth = None
        if truths is not None:
            item = truths[name]
            truth = np.asarray(item, bool) if isinstance(item, np.ndarray) else load_mask(item, image.shape)
        for variant in variants:
            trial_config = get_config(**{**cfg, **variant["overrides"]})
            result = preprocess_image(image, trial_config, geometry=geometry)
            row = {"source": str(name), "variant": variant["id"],
                   "preprocessing_seconds": result["metadata"]["preprocessing_seconds"],
                   "candidate_pixels": int(result["candidates"].sum())}
            if truth is not None:
                from .candidate_evaluation import scoped_mask_metrics
                row.update(scoped_mask_metrics(result["candidates"], truth))
            rows.append(row)
            if root:
                from .export import save_image_result
                save_image_result(result, root / variant["id"] / stem, trial_config,
                                  source_path=None if isinstance(source, np.ndarray) else source)
    aggregate, best = [], None
    if truths is not None:
        from .reporting import aggregate_trials, select_best
        aggregate = aggregate_trials(rows)
        best = select_best(aggregate)
    if root:
        write_json(root / "per_image_metrics.json", rows)
        write_json(root / "aggregate_metrics.json", aggregate)
        if rows:
            with (root / "per_image_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        if best:
            variant = next(v for v in variants if v["id"] == best["variant"])
            write_json(root / "best_variant.json", {**variant, "metrics": best,
                       "parameters": get_config(**{**cfg, **variant["overrides"]})})
    return {"per_image": rows, "aggregate": aggregate, "best": best}


def run_preview(images, **kwargs):
    """Section 9: run a single fixed setting without conducting a parameter sweep."""
    return run_sweep(images, [{"id": "preview", "overrides": {}}], **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("preview", "laplacian", "normalization", "clahe", "bilateral", "bilateral-location"))
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--preset", choices=("baseline", "recommended"), default="recommended")
    parser.add_argument("--truth", help="Optional binary ground-truth union PNG")
    parser.add_argument("--config", help="Config JSON, or notebook export with parameters")
    args = parser.parse_args()
    from .image_io import read_json
    overrides = read_json(args.config) if args.config else {}
    cfg = get_config(args.preset, **overrides.get("parameters", overrides))
    factories = {"preview": lambda: [{"id": "preview", "overrides": {}}],
                 "laplacian": laplacian_variants,
                 "normalization": lambda: normalization_variants(cfg.get("post_clahe_normalization_percentiles", cfg["normalization_percentiles"])),
                 "clahe": lambda: clahe_variants(cfg["clahe_clip_limit"], cfg["clahe_grid"]),
                 "bilateral": bilateral_variants, "bilateral-location": bilateral_location_variants}
    report = run_sweep({"image": args.input}, factories[args.experiment](), config=cfg,
                       truths={"image": args.truth} if args.truth else None, output_dir=args.output)
    print(f"Saved {len(report['per_image'])} trials to {args.output}")
    if report["best"]:
        print(f"Best measured variant: {report['best']['variant']}")
