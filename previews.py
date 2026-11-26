"""Section 6: inspect intermediate images without recomputing filters."""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .normalization import robust_unit, to_uint8
from .laplacian_filter import signed_laplacian_view


def show_stages(record, result, cfg):
    mask = result["disk_mask"]
    disk = result["metadata"]["disk"]
    outlined = cv2.cvtColor(result["original"], cv2.COLOR_GRAY2RGB)
    cv2.circle(
        outlined,
        (round(disk["cx"]), round(disk["cy"])),
        round(disk["radius"]),
        (255, 100, 40),
        max(1, result["original"].shape[0] // 700),
    )

    def scaled(key):
        return to_uint8(
            robust_unit(result[key], mask, cfg["normalization_percentiles"])[0]
        )

    panels = [
        ("1 Original", result["original"]),
        ("2 Detected disk", outlined),
        ("3 Disk mask", mask),
        ("4 Masked", result["masked"]),
        ("5 Median 3 x 3", result["median"]),
        ("6 Radial corrected", scaled("radial")),
        ("7 Polynomial background", scaled("background")),
        ("8 Corrected", scaled("corrected")),
        ("9 CLAHE", result["clahe"]),
        ("10 Normalized enhanced", result["enhanced"]),
        ("11 Signed Laplacian", signed_laplacian_view(result["laplacian"], mask)),
        ("12 Threshold candidates", result["raw_candidates"]),
        ("13 Area filtered", result["filtered_candidates"]),
        ("14 Final candidates", result["candidates"]),
        ("15 Three-channel RGB", result["multi_channel"]),
    ]
    fig, axes = plt.subplots(3, 5, figsize=(18, 11), constrained_layout=True)
    for ax, (title, array) in zip(axes.flat, panels):
        ax.imshow(array, cmap="gray", vmin=0, vmax=1 if array.dtype == bool else 255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"{record['name']} | {disk['method']} | "
        f"{result['metadata']['preprocessing_seconds']:.2f} s",
        fontsize=13,
    )
    plt.show()
    plt.close(fig)
    metadata = result["metadata"]
    thresholds = metadata["threshold_selection"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3), constrained_layout=True)
    axes[0].plot(metadata["radial_centers"], metadata["radial_profile"])
    axes[0].set(
        xlabel="Normalized disk radius",
        ylabel="Median intensity",
        title="Radial profile",
    )
    plotted_edges = (
        thresholds["edges"][1:]
        if thresholds["threshold_edge"] == "upper"
        else thresholds["edges"][:-1]
    )
    axes[1].plot(
        plotted_edges,
        [np.nan if x is None else x for x in thresholds["mean_laplacian"]],
        marker=".",
    )
    if thresholds["threshold"] is not None:
        axes[1].axvline(
            thresholds["threshold"], color="tab:red", label="Selected threshold"
        )
        axes[1].legend()
    axes[1].set(
        xlabel="Candidate intensity threshold [0, 1]",
        ylabel="Mean signed Laplacian",
        title="Newly included pixels in 20 intervals",
    )
    plt.show()
    plt.close(fig)
