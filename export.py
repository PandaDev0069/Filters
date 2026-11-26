"""Section 8: lossless exports, fingerprints, and parallel dataset caching."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import platform
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from .config import PIPELINE_VERSION
from .image_io import (write_png, write_numpy, write_json, read_json, digest, file_sha256, load_grayscale)
from .normalization import robust_unit, to_uint8
from .laplacian_filter import LAPLACIAN_KERNEL, signed_laplacian_view
from .pipeline import preprocess_image


def save_image_result(result, output_dir, cfg, source_path=None):
    """Save one composable pipeline run, keeping every float/label array exact."""
    from .image_io import save_array

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for stage, array in result.items():
        if not isinstance(array, np.ndarray):
            continue
        suffix = ".png" if array.dtype in (np.dtype("uint8"), np.dtype("bool")) else ".npy"
        path = root / (stage + suffix)
        save_array(path, array)
        outputs.append(path.name)
    metadata = {**result["metadata"], "config": cfg, "outputs": outputs, "complete": True}
    if source_path is not None:
        metadata.update(source=str(Path(source_path).resolve()), source_sha256=file_sha256(source_path))
    write_json(root / "metadata.json", metadata)


def display(value):
    print(value.to_string(index=False) if hasattr(value, "to_string") else value)


def output_paths(record, cfg):
    name = record["output_name"]
    stem = Path(name).stem
    paths = [
        f"{stage}/{name}"
        for stage in (
            "original",
            "corrected",
            "enhanced",
            "laplacian",
            "candidates",
            "multi_channel",
        )
    ]
    paths.append(f"laplacian/{stem}.npy")
    if cfg["save_intermediates"]:
        paths += [
            f"{stage}/{name}"
            for stage in (
                "disk_mask",
                "masked",
                "median",
                "radial",
                "background",
                "clahe",
                "raw_candidates",
                "filtered_candidates",
            )
        ]
    if cfg["save_float_stages"]:
        paths.append(f"float_stages/{stem}.npz")
    return paths


def save_processed(record, result, root, cfg, source_hash):
    directory = Path(root) / record["split"]
    name, stem = record["output_name"], Path(record["output_name"]).stem
    disk_mask = result["disk_mask"]
    stages = ["original", "corrected", "enhanced", "candidates", "multi_channel"]
    if cfg["save_intermediates"]:
        stages += [
            "disk_mask",
            "masked",
            "median",
            "radial",
            "background",
            "clahe",
            "raw_candidates",
            "filtered_candidates",
        ]
    for stage in stages:
        array = result[stage]
        if array.dtype == bool:
            array = array.astype(np.uint8) * 255
        elif array.dtype != np.uint8:
            array = to_uint8(
                robust_unit(array, disk_mask, cfg["normalization_percentiles"])[0]
            )
        write_png(directory / stage / name, array, rgb=stage == "multi_channel")
    write_numpy(
        directory / "laplacian" / (stem + ".npy"), laplacian=result["laplacian"]
    )
    write_png(
        directory / "laplacian" / name,
        signed_laplacian_view(result["laplacian"], disk_mask),
    )
    if cfg["save_float_stages"]:
        write_numpy(
            directory / "float_stages" / (stem + ".npz"),
            radial=result["radial"],
            background=result["background"],
            corrected=result["corrected"],
        )
    metadata = {
        **result["metadata"],
        "source": record["source"],
        "source_sha256": source_hash,
        "image_ids": record["image_ids"],
        "pipeline_sha256": digest({"version": PIPELINE_VERSION, "config": cfg}),
        "outputs": output_paths(record, cfg),
        "complete": True,
    }
    write_json(directory / "metadata" / (stem + ".json"), metadata)
    return metadata


def cache_record(record, input_dir, output_dir, cfg):
    started = time.perf_counter()
    path = Path(input_dir) / record["source"]
    source_hash = file_sha256(path)
    key = digest({"version": PIPELINE_VERSION, "config": cfg})
    directory = Path(output_dir) / record["split"]
    metadata_path = (
        directory / "metadata" / (Path(record["output_name"]).stem + ".json")
    )
    if metadata_path.exists():
        try:
            cached = read_json(metadata_path)
        except (ValueError, OSError):
            cached = {}
        if (
            cached.get("complete")
            and cached.get("source_sha256") == source_hash
            and cached.get("pipeline_sha256") == key
            and cached.get("outputs") == output_paths(record, cfg)
            and all(
                (directory / p).is_file() and (directory / p).stat().st_size > 0
                for p in output_paths(record, cfg)
            )
        ):
            return dict(
                source=record["source"],
                split=record["split"],
                status="cached",
                preprocessing_seconds=cached["preprocessing_seconds"],
                total_seconds=time.perf_counter() - started,
            )
    image = load_grayscale(path, cfg, record.get("coco_shape"))
    result = preprocess_image(image, cfg)
    save_processed(record, result, output_dir, cfg, source_hash)
    return dict(
        source=record["source"],
        split=record["split"],
        status="processed",
        preprocessing_seconds=result["metadata"]["preprocessing_seconds"],
        total_seconds=time.perf_counter() - started,
    )


def export_coco_subset(coco, successful_records, path):
    names_by_id = {
        image_id: record["output_name"]
        for record in successful_records
        for image_id in record["image_ids"]
    }
    subset = {
        key: value
        for key, value in coco.items()
        if key not in ("images", "annotations")
    }
    subset["images"] = [
        dict(row, file_name=names_by_id[row["id"]])
        for row in coco["images"]
        if row["id"] in names_by_id
    ]
    subset["annotations"] = [
        ann for ann in coco["annotations"] if ann["image_id"] in names_by_id
    ]
    write_json(path, subset)


def export_dataset(records, input_dir, output_dir, cfg, coco, splits, workers=2):
    if not set(splits) <= {"train", "val", "test"}:
        raise ValueError("EXPORT_SPLITS must contain train, val, or test.")
    selected = [r for r in records if r["split"] in splits]
    successes, errors = [], []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        jobs = {
            executor.submit(cache_record, r, input_dir, output_dir, cfg): r
            for r in selected
        }
        for future in tqdm(
            as_completed(jobs), total=len(jobs), desc="Preprocessing/export"
        ):
            record = jobs[future]
            try:
                successes.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "source": record["source"],
                        "split": record["split"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    successes.sort(key=lambda r: r["source"])
    errors.sort(key=lambda r: r["source"])
    complete_sources = {r["source"] for r in successes}
    complete_records = [r for r in selected if r["source"] in complete_sources]
    for split in splits:
        subset = [r for r in complete_records if r["split"] == split]
        if split != "test":
            export_coco_subset(
                coco, subset, Path(output_dir) / split / "annotations.json"
            )
        write_json(Path(output_dir) / split / "image_manifest.json", subset)
    write_json(Path(output_dir) / "export_errors.json", errors)
    write_json(Path(output_dir) / "export_summary.json", successes)
    print(f"Cached {len(successes)}/{len(selected)} images; {len(errors)} errors.")
    if errors:
        display(pd.DataFrame(errors))
    return successes, errors


def save_run_config(path, cfg, manifest, annotation_path):
    from .laplacian_filter import effective_signed_kernel
    spec = cfg.get("signed_laplacian_variant") or {}
    kernel = effective_signed_kernel({"kernel": spec.get("raw_kernel", spec.get("kernel", LAPLACIAN_KERNEL)),
                                      "gain": spec.get("gain", 1)})
    write_json(
        path,
        {
            "pipeline_version": PIPELINE_VERSION,
            "parameters": cfg,
            "pipeline_sha256": digest({"version": PIPELINE_VERSION, "config": cfg}),
            "split_manifest_sha256": digest(manifest),
            "annotation_sha256": file_sha256(annotation_path),
            "seed": manifest["seed"],
            "val_fraction": manifest["val_fraction"],
            "normalization_scope": "per-image on-disk percentiles; no dataset-wide fit",
            "channel_order_rgb": ["original", "laplacian_input", "candidates"],
            "laplacian_kernel": kernel.tolist(),
            "closing_enabled": bool(cfg["closing_kernel"]),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "pillow": Image.__version__,
            },
        },
    )
