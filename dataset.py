"""Section 2: discover physical images and freeze splits by observation date."""

from pathlib import Path
from collections import defaultdict
import re
import numpy as np
import pandas as pd
from PIL import Image
from .config import ANNOTATION_NAME, IMAGE_EXTENSIONS
from .image_io import read_json, write_json, digest

ANNOTATOR_COHORT_ORDER = (3, 2, 1)


def find_image_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def date_group(name, image_rows=()):
    match = re.match(r"(\d{8})", Path(name).name)
    if match:
        return match.group(1)
    dates = {
        str(row.get("date_captured", ""))[:10].replace("-", "")
        for row in image_rows
        if row.get("date_captured")
    }
    if len(dates) == 1:
        return dates.pop()
    raise ValueError(
        f"Cannot assign one observation date to {name}; define its date group."
    )


def discover_records(input_dir, coco):
    input_dir = Path(input_dir)
    rows_by_name = defaultdict(list)
    ids = set()
    for row in coco["images"]:
        if row["id"] in ids:
            raise ValueError(f"Duplicate COCO image id: {row['id']}")
        ids.add(row["id"])
        rows_by_name[Path(row["file_name"]).name].append(row)
    records = []
    for source_split in ("train", "test"):
        paths = find_image_files(input_dir / source_split)
        seen_names, seen_stems = set(), set()
        for path in paths:
            if path.name in seen_names or path.stem in seen_stems:
                raise ValueError(
                    f"Ambiguous output filename in {source_split}: {path.name}"
                )
            seen_names.add(path.name)
            seen_stems.add(path.stem)
            rows = rows_by_name.get(path.name, []) if source_split == "train" else []
            if source_split == "train" and not rows:
                raise ValueError(
                    f"No COCO image metadata for training file {path.name}"
                )
            shapes = {(int(r["height"]), int(r["width"])) for r in rows}
            if len(shapes) > 1:
                raise ValueError(f"Conflicting image dimensions for {path.name}")
            records.append(
                {
                    "source": path.relative_to(input_dir).as_posix(),
                    "name": path.name,
                    "output_name": path.stem + ".png",
                    "source_split": source_split,
                    "group": (
                        date_group(path.name, rows) if source_split == "train" else None
                    ),
                    "image_ids": [r["id"] for r in rows],
                    "coco_shape": list(next(iter(shapes))) if shapes else None,
                }
            )
    train_names = {r["name"] for r in records if r["source_split"] == "train"}
    test_names = {r["name"] for r in records if r["source_split"] == "test"}
    if train_names & test_names:
        raise ValueError(
            "Train and test share source filenames; resolve the overlap first."
        )
    if not train_names:
        raise ValueError("No training images found.")
    return records


def make_split_manifest(records, seed, val_fraction):
    if not 0 < val_fraction < 1:
        raise ValueError("VAL_FRACTION must be between zero and one.")
    groups = sorted({r["group"] for r in records if r["source_split"] == "train"})
    if len(groups) < 2:
        raise ValueError(
            "At least two observation dates are needed for a held-out split."
        )
    shuffled = np.random.default_rng(seed).permutation(groups)
    n_val = max(1, min(len(groups) - 1, round(len(groups) * val_fraction)))
    val_groups = set(shuffled[:n_val])
    result = []
    for record in records:
        split = (
            "test"
            if record["source_split"] == "test"
            else "val" if record["group"] in val_groups else "train"
        )
        result.append(dict(record, split=split))
    return {
        "seed": seed,
        "val_fraction": val_fraction,
        "inventory_sha256": digest(records),
        "records": result,
    }


def freeze_manifest(records, path, seed, val_fraction):
    proposed = make_split_manifest(records, seed, val_fraction)
    path = Path(path)
    if path.exists():
        existing = read_json(path)
        if existing != proposed:
            raise ValueError(
                "Saved split differs from current data/settings. Use a new OUTPUT_DIR."
            )
        return existing
    write_json(path, proposed)
    return proposed


def select_preview_records(records, input_dir, count):
    training = [r for r in records if r["split"] == "train"]
    ranked, errors = [], []
    for record in training:
        try:
            with Image.open(Path(input_dir) / record["source"]) as image:
                thumb = image.convert("L")
                thumb.thumbnail((128, 128))
                array = np.asarray(thumb)
            on_disk = array[array > max(10, float(np.percentile(array, 30)))]
            if not on_disk.size:
                raise ValueError("Empty foreground in preview thumbnail")
            contrast = float(np.percentile(on_disk, 90) - np.percentile(on_disk, 10))
            ranked.append((contrast, record["source"], record))
        except Exception as exc:
            errors.append({"source": record["source"], "error": str(exc)})
    if not ranked:
        raise ValueError(f"No readable training previews. Errors: {errors[:3]}")
    ranked.sort(key=lambda row: row[:2])
    indices = np.unique(
        np.linspace(0, len(ranked) - 1, min(count, len(ranked))).round().astype(int)
    )
    return [ranked[i][2] for i in indices], errors


def annotation_slot(image_id):
    prefix = str(image_id).split("-", 1)[0]
    slot = prefix[-2:]
    if slot not in {"01", "02", "03"}:
        raise ValueError(f"Unrecognized annotation-slot suffix: {image_id}")
    return slot


def select_annotator_cohorts(records, count, seed):
    if int(count) != count or count < 1:
        raise ValueError("Images per annotator cohort must be a positive integer.")
    selected, inventory = [], []
    for n_annotators in ANNOTATOR_COHORT_ORDER:
        available = sorted(
            [r for r in records if r["split"] == "train" and len(r["image_ids"]) == n_annotators],
            key=lambda r: r["source"],
        )
        if len(available) < count:
            raise ValueError(f"Need {count} training images with {n_annotators} annotators; found {len(available)}.")
        chosen = np.random.default_rng(seed + n_annotators).choice(len(available), size=int(count), replace=False)
        chosen_records = sorted([available[i] for i in chosen], key=lambda r: r["source"])
        for record in chosen_records:
            slots = [annotation_slot(image_id) for image_id in record["image_ids"]]
            if len(set(slots)) != n_annotators:
                raise ValueError(f"Duplicate annotation slots for {record['name']}.")
            selected.append(dict(record, annotator_count=n_annotators))
        inventory.append({"annotator_count": n_annotators, "available_training_images": len(available), "selected_images": len(chosen_records)})
    if len({r["source"] for r in selected}) != len(selected):
        raise ValueError("Annotator cohorts must contain distinct physical images.")
    return selected, pd.DataFrame(inventory)
