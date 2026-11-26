"""Section 20: connected candidate instances and class-agnostic PQ evaluation."""

import cv2
import numpy as np


def export_instances(labels, output_dir, image_name="image"):
    """Save exact labels, binary foreground, and one COCO-style RLE per instance."""
    from pathlib import Path
    from pycocotools import mask as coco_mask
    from .image_io import save_array, write_json

    labels = np.asarray(labels)
    if (labels.ndim != 2 or labels.dtype.kind not in "iu" or labels.min() < 0
            or not np.array_equal(np.unique(labels[labels > 0]), np.arange(1, int(labels.max()) + 1))):
        raise ValueError("Provide contiguous nonnegative integer instance labels.")
    root = Path(output_dir)
    save_array(root / "labels.npz", labels)
    save_array(root / "binary.png", labels > 0)
    rows = []
    for label_id in range(1, int(labels.max()) + 1):
        mask = (labels == label_id).astype(np.uint8)
        rle = coco_mask.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        x, y, width, height = cv2.boundingRect(mask)
        rows.append({"id": label_id, "image_id": image_name, "category_id": 1,
                     "segmentation": rle, "area": int(mask.sum()),
                     "bbox": [x, y, width, height], "iscrowd": 0})
    write_json(root / "instances.json", {"image": image_name,
               "shape": list(labels.shape), "annotations": rows,
               "categories": [{"id": 1, "name": "filament"}]})
    return rows


def candidates_to_instances(binary):
    binary = np.asarray(binary)
    if binary.ndim != 2 or not np.isin(binary, [0, 1]).all():
        raise ValueError("Expected a two-dimensional binary candidate mask.")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    assert np.array_equal(labels > 0, binary.astype(bool))
    return labels, stats[1:], count - 1


def match_instances(iou, threshold=0.5):
    """Maximum number of valid pairs, then maximum total IoU; no reused IDs."""
    iou = np.asarray(iou, dtype=float)
    if iou.ndim != 2 or not np.isfinite(iou).all() or np.any((iou < 0) | (iou > 1)):
        raise ValueError("Expected a finite GT-by-prediction IoU matrix in [0, 1].")
    if not 0 <= threshold < 1:
        raise ValueError("IoU threshold must lie in [0, 1).")
    if min(iou.shape) == 0:
        return []
    valid = iou > threshold
    # Losing one valid match costs more than any possible total-IoU improvement.
    bonus = min(iou.shape) + 1.0
    weight = np.where(valid, bonus + iou, 0.0)
    from scipy.optimize import linear_sum_assignment
    gt_ids, pred_ids = linear_sum_assignment(weight, maximize=True)
    return [(int(g), int(p), float(iou[g, p]))
            for g, p in zip(gt_ids, pred_ids) if valid[g, p]]


def instance_scores(tp, fp, fn, sum_iou):
    denominator = tp + 0.5 * (fp + fn)
    return {
        "pq": float(sum_iou / denominator) if denominator else 0.0,
        "sq": float(sum_iou / tp) if tp else 0.0,
        "rq": float(tp / denominator) if denominator else 0.0,
        "instance_precision": float(tp / (tp + fp)) if tp + fp else np.nan,
        "instance_recall": float(tp / (tp + fn)) if tp + fn else np.nan,
    }


def pixel_scores(tp, fp, fn, tn):
    return {
        "pixel_precision": float(tp / (tp + fp)) if tp + fp else np.nan,
        "pixel_recall": float(tp / (tp + fn)) if tp + fn else np.nan,
        "pixel_iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
        "pixel_dice": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.0,
    }


def evaluate_instances(labels, gt_masks):
    """Keep all original GT masks intact; use label histograms for intersections."""
    n_pred = int(labels.max())
    areas = np.bincount(labels.ravel(), minlength=n_pred + 1)[1:]
    iou = np.zeros((len(gt_masks), n_pred), dtype=float)
    truth = np.zeros(labels.shape, bool)
    coverage = np.zeros(labels.shape, np.uint16)
    for g, mask in enumerate(gt_masks):
        mask = np.asarray(mask, bool)
        if mask.shape != labels.shape or not mask.any():
            raise ValueError("GT instances must be nonempty and on the native image grid.")
        truth |= mask
        coverage += mask
        intersections = np.bincount(labels[mask], minlength=n_pred + 1)[1:]
        iou[g] = intersections / (int(mask.sum()) + areas - intersections)
    matches = match_instances(iou)
    tp, fp, fn = len(matches), n_pred - len(matches), len(gt_masks) - len(matches)
    sum_iou = sum(value for _, _, value in matches)
    pred = labels > 0
    pixel = {"pixel_tp": int(np.count_nonzero(pred & truth)),
             "pixel_fp": int(np.count_nonzero(pred & ~truth)),
             "pixel_fn": int(np.count_nonzero(~pred & truth)),
             "pixel_tn": int(np.count_nonzero(~pred & ~truth))}
    row = {"gt_instances": len(gt_masks), "pred_instances": n_pred,
           "instance_tp": tp, "instance_fp": fp, "instance_fn": fn,
           "sum_matched_iou": sum_iou, "gt_overlap_pixels": int((coverage > 1).sum()),
           **instance_scores(tp, fp, fn, sum_iou), **pixel,
           **pixel_scores(*(pixel[k] for k in ("pixel_tp", "pixel_fp", "pixel_fn", "pixel_tn")))}
    assert np.isclose(row["pq"], row["sq"] * row["rq"])
    return row, iou, matches, truth


def apply_instance_segmentation(input_path, output_path, config=None, **options):
    """File API for the instances stage; accepts mask_path/disk_path/reference_path."""
    from .cli import apply_stage
    return apply_stage("instances", input_path, output_path, config=config, **options)


if __name__ == "__main__":
    from .cli import main
    main(["instances", *__import__("sys").argv[1:]])
