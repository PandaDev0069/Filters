"""Reusable algorithms from notebook cell 26; no notebook execution required."""

import math
import cv2
import numpy as np


def fit_annotation_size_bounds(coco, records, minimum_quantile=0.01,
                               maximum_quantile=1.0, span_quantile=1.0):
    """Section 10: fit area/span rules using only frozen training records.

    Each original annotation instance is rasterized independently, including
    multiple annotation sets for a physical image. Validation/test masks do not fit
    the bounds. Return a rule accepted by the pipeline's annotation_area_filter.
    """
    from collections import defaultdict
    from .candidate_evaluation import decode_annotation

    if not 0 <= minimum_quantile <= maximum_quantile <= 1 or not 0 < span_quantile <= 1:
        raise ValueError("Require ordered area quantiles and a span quantile in (0, 1].")
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    areas, spans, measured, empty, crowd = [], [], 0, 0, 0
    seen = set()
    for record in records:
        if record["split"] != "train":
            continue
        for image_id in record["image_ids"]:
            if image_id in seen:
                raise ValueError("Each COCO image ID must occur once in training records.")
            seen.add(image_id)
            for annotation in annotations[image_id]:
                mask = decode_annotation(annotation, tuple(record["coco_shape"]))
                measured += 1
                area = int(mask.sum())
                empty += area == 0
                crowd += bool(annotation.get("iscrowd", False))
                if area == 0 or annotation.get("iscrowd", False):
                    continue
                _, _, width, height = cv2.boundingRect(mask.astype(np.uint8))
                areas.append(area)
                spans.append(max(width, height))
    if not areas:
        raise ValueError("No nonempty, non-crowd training instances available to fit bounds.")
    minimum = math.ceil(float(np.quantile(areas, minimum_quantile)))
    maximum = math.floor(float(np.quantile(areas, maximum_quantile)))
    span = math.floor(float(np.quantile(spans, span_quantile)))
    if minimum > maximum:
        raise ValueError("The quantiles do not define a valid integer area range.")
    return {
        "fit_split": "train", "measurement": "native-resolution rasterized annotation-instance area",
        "candidate_connectivity": 8, "quantile": minimum_quantile,
        "maximum_area_quantile": maximum_quantile, "maximum_span_quantile": span_quantile,
        "observed_minimum_area_px": min(areas), "observed_maximum_area_px": max(areas),
        "minimum_candidate_area_px": minimum, "maximum_candidate_area_px": maximum,
        "maximum_candidate_bbox_span_px": span, "measured_instances": measured,
        "empty_masks_excluded": empty, "crowd_annotations_excluded": crowd,
        "training_instance_size_retention": float(np.mean(
            (np.asarray(areas) >= minimum) & (np.asarray(areas) <= maximum)
            & (np.asarray(spans) <= span))),
    }


def filter_candidate_size_bounds(mask, minimum_area_px, maximum_area_px, maximum_span_px):
    minimum, maximum, max_span = int(minimum_area_px), int(maximum_area_px), int(maximum_span_px)
    if (minimum != minimum_area_px or maximum != maximum_area_px or max_span != maximum_span_px
            or not 1 <= minimum <= maximum or max_span < 1):
        raise ValueError("Size bounds must be positive whole pixels with minimum area <= maximum area.")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    spans = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    too_small = areas < minimum
    too_large = areas > maximum
    too_wide = spans > max_span
    keep = ~(too_small | too_large | too_wide)
    keep[0] = False
    filtered = keep[labels]
    # Slice away the background label for every diagnostic count.
    return filtered, {
        "minimum_area_px": minimum,
        "maximum_area_px": maximum,
        "maximum_bbox_span_px": max_span,
        "components_before": count - 1,
        "components_after": int(keep.sum()),
        "components_removed": count - 1 - int(keep.sum()),
        "undersized_components_removed": int(too_small[1:].sum()),
        "oversized_area_components_removed": int(too_large[1:].sum()),
        "excessive_span_components_removed": int(too_wide[1:].sum()),
        "upper_bound_components_removed": int((too_large[1:] | too_wide[1:]).sum()),
        "pixels_removed": int(np.count_nonzero(mask) - np.count_nonzero(filtered)),
    }


def apply_annotation_size_filter(input_path, output_path, config=None, **options):
    """File API for the size stage; accepts mask_path/disk_path/reference_path."""
    from .cli import apply_stage
    return apply_stage("size", input_path, output_path, config=config, **options)


if __name__ == "__main__":
    from .cli import main
    main(["size", *__import__("sys").argv[1:]])
