"""Section 19: pooled scorecards and stable selection across saved trials."""

import math
from .candidate_evaluation import extended_pixel_metrics
from .image_io import read_json, write_json


def aggregate_trials(rows):
    """Pool confusion counts per variant, retaining first-seen variant order."""
    groups = {}
    for row in rows:
        groups.setdefault(row["variant"], []).append(row)
    output = []
    for variant, items in groups.items():
        counts = {key: sum(int(row[key]) for row in items) for key in ("tp", "fp", "fn", "tn")}
        output.append({"variant": variant, "images": len(items),
                       **extended_pixel_metrics(**counts)})
    return output


def select_best(rows):
    """Highest pooled IoU, then Dice; original order wins exact ties."""
    if not rows:
        raise ValueError("No measured variants available.")

    def score(row):
        return tuple(float(row[key]) if row.get(key) is not None and math.isfinite(row[key])
                     else -math.inf for key in ("iou", "dice"))

    return max(rows, key=score)


def summarize_saved_sweep(path):
    """Rebuild a section-19-style scorecard from a sweep's per-image JSON."""
    rows = aggregate_trials(read_json(path))
    return rows, select_best(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", help="per_image_metrics.json from a measured sweep")
    parser.add_argument("output", help="Output scorecard JSON")
    args = parser.parse_args()
    rows, best = summarize_saved_sweep(args.metrics)
    write_json(args.output, {"aggregate": rows, "best": best})
    print(f"Selected {best['variant']} by pooled IoU, then Dice.")
