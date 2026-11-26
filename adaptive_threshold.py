"""Section 5: score intensity intervals with the signed Laplacian response."""

import cv2
import numpy as np
from .config import get_config
from .laplacian_filter import signed_laplacian


def candidate_region(disk_mask, margin=2):
    """Exclude the immediate limb with repeated 3x3 binary erosion."""
    if int(margin) != margin or margin < 0:
        raise ValueError("Candidate limb margin must be a nonnegative integer.")
    mask = np.asarray(disk_mask, bool)
    if mask.ndim != 2:
        raise ValueError("Disk mask must be two-dimensional.")
    return (cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                      iterations=int(margin)).astype(bool) if margin else mask.copy())


def select_candidates(enhanced, laplacian, eligible, intervals=20,
                      sigma_span=3.0, threshold_edge="upper"):
    """Tune thresholding independently using an already computed response.

    Scores use newly included pixels in each interval. The final mask thresholds
    enhanced intensity, rather than thresholding the Laplacian itself.
    """
    if enhanced.dtype != np.uint8 or enhanced.ndim != 2:
        raise ValueError("Enhanced image must be two-dimensional uint8.")
    if laplacian.shape != enhanced.shape or eligible.shape != enhanced.shape:
        raise ValueError("Response and eligible mask must match the enhanced image.")
    if int(intervals) != intervals or intervals < 2:
        raise ValueError("Use at least two whole threshold intervals.")
    if not np.isfinite(sigma_span) or sigma_span <= 0:
        raise ValueError("Threshold sigma span must be finite and positive.")
    if threshold_edge not in ("upper", "lower"):
        raise ValueError("threshold_edge must be upper or lower.")
    eligible = np.asarray(eligible, bool)
    unit = enhanced.astype(np.float32) / 255.0
    values = unit[eligible]
    if not values.size:
        raise ValueError("Candidate mask has no interior pixels.")
    if not np.isfinite(laplacian[eligible]).all():
        raise ValueError("Signed response must be finite inside the eligible region.")
    median, std = float(np.median(values)), float(np.std(values))
    edges = np.linspace(median - sigma_span * std, median, int(intervals) + 1)
    scores = np.full(int(intervals), np.nan)
    counts = np.zeros(int(intervals), np.int64)
    responses = laplacian[eligible]
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        included = (values >= low) & (values < high)
        counts[i] = included.sum()
        if counts[i]:
            scores[i] = float(np.mean(responses[included]))
    selected = None
    raw = np.zeros(enhanced.shape, bool)
    if std > 1e-6 and np.isfinite(scores).any():
        best = int(np.nanargmax(scores))
        if scores[best] > 0:
            selected = float(edges[best + (threshold_edge == "upper")])
            raw = eligible & (unit < selected)
    return raw, {
        "median": median, "std": std, "threshold": selected,
        "threshold_edge": threshold_edge, "edges": edges.tolist(),
        "counts": counts.tolist(),
        "mean_laplacian": [float(s) if np.isfinite(s) else None for s in scores],
    }


def adaptive_candidates(enhanced, disk_mask, cfg=None, kernel=None):
    """Notebook-compatible combined response and candidate extraction."""
    cfg = get_config(**(cfg or {}))
    laplacian = signed_laplacian(enhanced, kernel)
    eligible = candidate_region(disk_mask, cfg["candidate_limb_margin_px"])
    laplacian[~eligible] = 0
    raw, diagnostics = select_candidates(
        enhanced, laplacian, eligible, cfg["threshold_intervals"],
        cfg["threshold_sigma_span"], cfg["threshold_edge"],
    )
    return laplacian, eligible, raw, diagnostics


def apply_adaptive_threshold(input_path, output_path, config=None, **options):
    """File API: enhanced image plus optional disk mask to raw candidates."""
    from .cli import apply_stage
    return apply_stage("threshold", input_path, output_path, config=config, **options)


if __name__ == "__main__":
    from .cli import main
    main(["threshold", *__import__("sys").argv[1:]])
