"""Reusable algorithms from notebook cell 17; no notebook execution required."""

import copy
import math
from collections import defaultdict
import cv2
import numpy as np
from pycocotools import mask as coco_mask


def evaluate_annotators(prediction, annotations_by_image, image_ids, disk_mask=None):
    """Section 12: evaluate each COCO image ID separately; never union annotators."""
    rows, truths = [], {}
    for image_id in image_ids:
        record = {"source_split": "train", "split": "train", "source": str(image_id),
                  "image_ids": [image_id]}
        result = {"candidates": np.asarray(prediction, bool),
                  "metadata": {"preprocessing_seconds": 0.0}}
        metrics, truth = evaluate_candidates(record, result, annotations_by_image)
        row = {"image_id": image_id, **metrics, **scoped_mask_metrics(prediction, truth)}
        if disk_mask is not None:
            row["disk_metrics"] = scoped_mask_metrics(prediction, truth, disk_mask)
        rows.append(row)
        truths[image_id] = truth
    return rows, truths


def decode_annotation(annotation, shape):
    height, width = shape
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros(shape, bool)
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segmentation, dict):
        rle = copy.deepcopy(segmentation)
        if list(rle["size"]) != [height, width]:
            raise ValueError("RLE dimensions disagree with source image.")
        if isinstance(rle["counts"], list):
            rle = coco_mask.frPyObjects(rle, height, width) # type: ignore
        elif isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
    else:
        raise ValueError("Unsupported COCO segmentation format.")
    decoded = coco_mask.decode(rle) # type: ignore
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=-1)
    if decoded.shape != tuple(shape):
        raise ValueError("Decoded annotation has incorrect dimensions.")
    return decoded.astype(bool)


def pixel_metrics(tp, fp, fn):
    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    return {
        "candidate_precision": ratio(tp, tp + fp),
        "candidate_recall": ratio(tp, tp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
    }


def evaluate_candidates(record, result, annotations_by_image):
    if record["source_split"] != "train":
        raise ValueError("Candidate evaluation requires supplied training annotations.")
    candidates = result["candidates"]
    n_components, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates.astype(np.uint8), connectivity=8
    )
    component_areas = stats[:, cv2.CC_STAT_AREA]
    truth = np.zeros(candidates.shape, bool)
    component_links = defaultdict(set)
    gt_links = defaultdict(set)
    pairs = []
    n_truth = 0
    for image_id in record["image_ids"]:
        for annotation in annotations_by_image.get(image_id, []):
            mask = decode_annotation(annotation, candidates.shape)
            area = int(mask.sum())
            truth |= mask
            if area == 0 or annotation.get("iscrowd", 0):
                continue
            gt_index = n_truth
            n_truth += 1
            intersections = np.bincount(labels[mask], minlength=n_components)
            for component in np.flatnonzero(intersections[1:]) + 1:
                intersection = int(intersections[component])
                if intersection / area >= 0.10:
                    component_links[int(component)].add(gt_index)
                    gt_links[gt_index].add(int(component))
                iou = intersection / (
                    area + int(component_areas[component]) - intersection
                )
                if iou > 0.5:
                    pairs.append((float(iou), gt_index, int(component)))
    used_gt, used_components, iou_sum = set(), set(), 0.0
    for iou, gt_index, component in sorted(pairs, reverse=True):
        if gt_index not in used_gt and component not in used_components:
            used_gt.add(gt_index)
            used_components.add(component)
            iou_sum += iou
    matched = len(used_gt)
    pred_count = n_components - 1
    denominator = matched + 0.5 * (pred_count - matched) + 0.5 * (n_truth - matched)
    tp = int(np.count_nonzero(candidates & truth))
    fp = int(np.count_nonzero(candidates & ~truth))
    fn = int(np.count_nonzero(~candidates & truth))
    metrics = {
        "source": record["source"],
        "split": record["split"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": int(np.count_nonzero(~candidates & ~truth)),
        **pixel_metrics(tp, fp, fn),
        "gt_instances": n_truth,
        "candidate_components": pred_count,
        "split_gt_instances": sum(len(v) > 1 for v in gt_links.values()),
        "merged_components": sum(len(v) > 1 for v in component_links.values()),
        "component_pq_diagnostic": iou_sum / denominator if denominator else None,
        "preprocessing_seconds": result["metadata"]["preprocessing_seconds"],
    }
    return metrics, truth


def candidate_confusion_image(prediction, truth):
    prediction, truth = np.asarray(prediction, dtype=bool), np.asarray(truth, dtype=bool)
    if prediction.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("Candidate and truth masks must have the same 2-D shape.")
    states = {
        "TP": truth & prediction,
        "FP": prediction & ~truth,
        "FN": truth & ~prediction,
        "TN": ~truth & ~prediction,
    }
    colors = {"TP": [50, 220, 80], "FP": [70, 140, 255], "FN": [255, 80, 80], "TN": [0, 0, 0]}
    overlap = np.zeros((*truth.shape, 3), np.uint8)
    for name, mask in states.items():
        overlap[mask] = colors[name]
    counts = {name: int(np.count_nonzero(mask)) for name, mask in states.items()}
    return overlap, counts, colors


def extended_pixel_metrics(tp, fp, fn, tn):
    tp, fp, fn, tn = map(int, (tp, fp, fn, tn))
    if min(tp, fp, fn, tn) < 0:
        raise ValueError("Confusion counts cannot be negative.")

    def ratio(a, b):
        return float(a / b) if b else None

    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    mcc_denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        **pixel_metrics(tp, fp, fn),
        "specificity": specificity,
        "false_positive_rate": ratio(fp, fp + tn),
        "false_negative_rate": ratio(fn, fn + tp),
        "accuracy": ratio(tp + tn, tp + fp + fn + tn),
        "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "mcc": (tp * tn - fp * fn) / math.sqrt(mcc_denominator) if mcc_denominator else None,
        "truth_foreground_fraction": ratio(tp + fn, tp + fp + fn + tn),
        "candidate_foreground_fraction": ratio(tp + fp, tp + fp + fn + tn),
    }


def scoped_mask_metrics(prediction, truth, scope=None):
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    if prediction.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("Prediction and annotation masks must share a 2-D shape.")
    valid = np.ones(prediction.shape, bool) if scope is None else np.asarray(scope, bool)
    if valid.shape != prediction.shape:
        raise ValueError("Evaluation scope must match the masks.")
    return extended_pixel_metrics(
        np.count_nonzero(valid & prediction & truth), np.count_nonzero(valid & prediction & ~truth),
        np.count_nonzero(valid & ~prediction & truth), np.count_nonzero(valid & ~prediction & ~truth),
    )
