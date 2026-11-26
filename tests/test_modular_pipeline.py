"""Behavioral tests against notebook definitions, rather than duplicated formulas."""

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest
from PIL import Image

from Filters.config import get_config
from Filters.pipeline import preprocess_image
from Filters.solar_disk_filter import detect_solar_disk

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def reference():
    nb = json.loads((ROOT / "Final/solar-filament-pre-processing.ipynb").read_text(encoding="utf-8"))
    names = {
        "resize_for_detection", "valid_circle", "detect_solar_disk", "robust_unit", "to_uint8",
        "radial_correction", "polynomial_correction", "enhance_image", "adaptive_candidates",
        "filter_components", "conservative_closing", "preprocess_image", "filter_candidate_size_bounds",
        "apply_annotation_area_filter", "radius_edge_geometry", "reject_radius_edge_components",
        "apply_radius_edge_filter", "adaptive_candidates_with_signed_kernel",
        "normalized_enhanced_from_clahe", "prepare_fixed_clahe_input", "apply_clahe_setting",
        "apply_bilateral_setting", "prepare_bilateral_location_shared", "location_bilateral",
        "build_bilateral_location",
    }
    namespace = {"np": np, "cv2": cv2, "math": math, "copy": copy, "time": time, "hashlib": hashlib,
                 "LAPLACIAN_KERNEL": np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook-reference", "exec"), namespace)
    return namespace


@pytest.fixture
def small_config():
    return dict(expected_shape=[256, 256], hough_coarse_size=256, hough_fine_size=256,
                radial_bins=32, radial_sample_stride=2, polynomial_sample_stride=4)


def synthetic_sun():
    y, x = np.ogrid[:256, :256]
    rho = np.hypot(x - 127.5, y - 127.5) / 110
    image = np.where(rho <= 1, 190 - 55 * rho ** 2 + 8 * x / 256, 0)
    image[(abs(y - 130) < 2) & (x > 80) & (x < 180)] -= 70
    return np.clip(image, 0, 255).astype(np.uint8)


def test_baseline_is_pixel_identical_to_notebook(reference, small_config):
    cfg = get_config(**small_config)
    image = synthetic_sun()
    before = image.copy()
    expected = reference["preprocess_image"](image, cfg)
    actual = preprocess_image(image, cfg)
    for key, array in expected.items():
        if isinstance(array, np.ndarray):
            np.testing.assert_array_equal(actual[key], array, err_msg=key)
    assert actual["metadata"]["threshold_selection"] == expected["metadata"]["threshold_selection"]
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("position", ["none", "before_median", "after_median", "before_clahe", "after_normalization"])
def test_selected_pipeline_and_each_bilateral_position_match_notebook(reference, small_config, position):
    cfg = get_config("recommended", **small_config)
    setting = {"id": "d5_c15_s3", "enabled": True, "diameter": 5, "sigma_color": 15, "sigma_space": 3}
    reference.update({
        "location_fixed_config": cfg,
        "location_fixed_clahe": {"clip_limit": cfg["clahe_clip_limit"], "tile_grid": cfg["clahe_grid"]},
        "location_fixed_normalization": {"percentiles": cfg["post_clahe_normalization_percentiles"]},
        "location_fixed_bilateral": setting,
        "location_fixed_kernel": reference["LAPLACIAN_KERNEL"],
        "location_fixed_laplacian": cfg["signed_laplacian_variant"],
        "location_fixed_size_rule": cfg["annotation_area_filter"],
        "location_fixed_radius_config": cfg["radius_edge_filter"],
        "BILATERAL_LOCATION_VARIANTS": [{"id": p} for p in ["none", "before_median", "after_median", "before_clahe", "after_normalization"]],
        "ANNOTATION_AREA_QUANTILE": 0.01, "ANNOTATION_MAX_AREA_QUANTILE": 1.0,
        "ANNOTATION_MAX_SPAN_QUANTILE": 1.0,
    })
    image = synthetic_sun()
    shared = reference["prepare_bilateral_location_shared"](image)
    expected, _ = reference["build_bilateral_location"](shared, {"id": position})
    cfg["bilateral_filter"] = {**setting, "enabled": position != "none", "position": position}
    actual = preprocess_image(image, cfg)
    for stage in ("median", "radial", "background", "corrected", "clahe_input", "clahe", "enhanced",
                  "laplacian_input", "laplacian", "raw_candidates", "filtered_candidates", "candidates", "multi_channel"):
        np.testing.assert_array_equal(actual[stage], expected[stage], err_msg=f"{position}: {stage}")
    assert not np.any(actual["candidates"] & ~actual["edge_safe_disk_mask"])


def test_lossless_independent_stages_compose_to_same_result(tmp_path, small_config):
    from Filters.cli import apply_stage
    from Filters.image_io import save_array, load_array
    cfg = get_config("recommended", **small_config)
    image = synthetic_sun()
    original = tmp_path / "original.png"
    save_array(original, image)
    mask = tmp_path / "disk.png"
    apply_stage("disk", original, mask, config=cfg)
    disk = tmp_path / "disk.png.json"
    steps = [("median", "median.png"), ("radial", "radial.npy"),
             ("polynomial", "corrected.npy"), ("prepare-clahe", "clahe_input.png"),
             ("clahe", "clahe.png"), ("normalize", "enhanced.png"),
             ("threshold", "raw_candidates.png"), ("components", "filtered.png"),
             ("closing", "closed.png"), ("size", "size.png"), ("radius", "candidates.png")]
    source = original
    for stage, filename in steps:
        destination = tmp_path / filename
        apply_stage(stage, source, destination, config=cfg, mask_path=mask, disk_path=disk)
        source = destination
    expected = preprocess_image(image, cfg)
    np.testing.assert_array_equal(load_array(tmp_path / "enhanced.png"), expected["enhanced"])
    np.testing.assert_array_equal(load_array(source) > 0, expected["candidates"])
    np.testing.assert_array_equal(load_array(tmp_path / "radial.npy"), expected["radial"])
    with pytest.raises(ValueError, match="Use .png"):
        save_array(tmp_path / "float.png", expected["radial"])


def test_size_fitting_excludes_validation_annotations():
    from pycocotools import mask as coco_mask
    from Filters.annotation_size_filter import fit_annotation_size_bounds
    annotations = []
    for image_id, side in [("train", 3), ("val", 15)]:
        mask = np.zeros((20, 20), np.uint8)
        mask[1:1 + side, 1:1 + side] = 1
        rle = coco_mask.encode(np.asfortranarray(mask))
        annotations.append({"id": image_id, "image_id": image_id, "segmentation": rle})
    records = [{"split": split, "image_ids": [split], "coco_shape": [20, 20]} for split in ["train", "val"]]
    rule = fit_annotation_size_bounds({"annotations": annotations}, records)
    assert rule["minimum_candidate_area_px"] == rule["maximum_candidate_area_px"] == 9
    assert rule["maximum_candidate_bbox_span_px"] == 3


def test_sweep_variants_are_independent_and_gain_controls_tie(small_config):
    from Filters.experiments import run_sweep, laplacian_variants
    cfg = get_config(**small_config)
    before = copy.deepcopy(cfg)
    image = synthetic_sun()
    baseline = preprocess_image(image, cfg)
    variants = laplacian_variants()
    report = run_sweep({"sun": image}, [variants[0], variants[-2], variants[-1]],
                       config=cfg, truths={"sun": baseline["candidates"]})
    assert cfg == before
    assert report["best"]["variant"] == "baseline_8n"
    assert all(row["iou"] == 1 for row in report["aggregate"])


def test_instances_preserve_binary_mask_and_match_overlapping_truth():
    from Filters.instance_segmentation import candidates_to_instances, evaluate_instances, match_instances
    binary = np.zeros((32, 32), bool)
    binary[2:8, 2:8] = True
    binary[20:25, 20:25] = True
    labels, _, count = candidates_to_instances(binary)
    np.testing.assert_array_equal(labels > 0, binary)
    row, _, matches, _ = evaluate_instances(labels, [labels == 1, labels == 2])
    assert count == 2 and len(matches) == 2 and row["pq"] == 1
    # Maximum cardinality takes precedence over greedily selecting the 0.9 pair.
    assert len(match_instances(np.array([[0.9, 0.6], [0.6, 0.0]]))) == 2


def test_module_cli_runs_without_notebook_or_dataset_discovery(tmp_path, small_config):
    source = tmp_path / "sun.png"
    Image.fromarray(synthetic_sun()).save(source)
    output = tmp_path / "median.png"
    command = [sys.executable, "-m", "Filters", "median", str(source), str(output),
               "--set", "strict_shape=false", "--set", "median_kernel=5"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(np.asarray(Image.open(output)), cv2.medianBlur(synthetic_sun(), 5))


def test_presets_do_not_share_nested_state():
    config = get_config("recommended")
    config["clahe_grid"][0] = 99
    assert get_config("recommended")["clahe_grid"] == [4, 4]
    assert get_config()["clahe_grid"] == [8, 8]
