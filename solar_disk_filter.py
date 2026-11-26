"""Reusable algorithms from notebook cell 9; no notebook execution required."""

import math
import cv2
import numpy as np


def resize_for_detection(image, long_side):
    scale = min(1.0, long_side / max(image.shape))
    size = (
        max(1, round(image.shape[1] * scale)),
        max(1, round(image.shape[0] * scale)),
    )
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    # Separate scale factors account for integer rounding on non-square inputs.
    return small, np.array([size[0] / image.shape[1], size[1] / image.shape[0]])


def valid_circle(cx, cy, radius, shape, cfg):
    h, w = shape
    lo, hi = cfg["disk_radius_fraction"]
    return (
        lo * min(h, w) <= radius <= hi * min(h, w)
        and abs(cx - (w - 1) / 2) < 0.20 * w
        and abs(cy - (h - 1) / 2) < 0.20 * h
        and cx - radius >= 0
        and cx + radius < w
        and cy - radius >= 0
        and cy + radius < h
    )


def detect_solar_disk(image, cfg):
    if float(np.std(image)) < 1:
        raise ValueError("Solar disk detection failed: image has almost no contrast.")
    coarse, scale = resize_for_detection(image, cfg["hough_coarse_size"])
    blur = cv2.medianBlur(coarse, 5)
    lo, hi = cfg["disk_radius_fraction"]
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(blur.shape) / 2,
        param1=cfg["hough_param1"],
        param2=cfg["hough_param2"],
        minRadius=round(lo * min(blur.shape)),
        maxRadius=round(hi * min(blur.shape)),
    )
    disk = None
    if circles is not None:
        for cx, cy, radius in circles[0]:
            values = (
                float(cx / scale[0]),
                float(cy / scale[1]),
                float(radius / scale.mean()),
            )
            if valid_circle(*values, image.shape, cfg):
                disk = dict(zip(("cx", "cy", "radius"), values), method="hough-coarse")
                break
    if disk is None:
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 20 and cv2.contourArea(contour) > 0.15 * blur.size:
                (cx, cy), (a, b), _ = cv2.fitEllipse(contour)
                values = (
                    float(cx / scale[0]),
                    float(cy / scale[1]),
                    float((a + b) / (4 * scale.mean())),
                )
                if min(a, b) / max(a, b) > 0.90 and valid_circle(
                    *values, image.shape, cfg
                ):
                    disk = dict(
                        zip(("cx", "cy", "radius"), values), method="contour-fallback"
                    )
    if disk is None:
        raise ValueError(
            "No plausible solar disk found; inspect image/detection settings."
        )

    if disk["method"].startswith("hough"):
        fine, fs = resize_for_detection(image, cfg["hough_fine_size"])
        r = disk["radius"] * fs.mean()
        tolerance = max(4, 0.025 * r)
        refined = cv2.HoughCircles(
            cv2.medianBlur(fine, 5),
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(fine.shape) / 2,
            param1=cfg["hough_param1"],
            param2=cfg["hough_param2"],
            minRadius=max(1, round(r - tolerance)),
            maxRadius=round(r + tolerance),
        )
        if refined is not None:
            for cx, cy, radius in refined[0]:
                values = (
                    float(cx / fs[0]),
                    float(cy / fs[1]),
                    float(radius / fs.mean()),
                )
                displacement = math.hypot(
                    values[0] - disk["cx"], values[1] - disk["cy"]
                )
                if displacement < 0.04 * disk["radius"] and valid_circle(
                    *values, image.shape, cfg
                ):
                    disk = dict(
                        zip(("cx", "cy", "radius"), values),
                        method="hough-coarse-to-fine",
                    )
                    break
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    rho = (np.hypot(xx - disk["cx"], yy - disk["cy"]) / disk["radius"]).astype(
        np.float32
    )
    mask = rho <= 1.0
    if np.median(image[mask]) <= np.median(image[~mask]) + 5:
        raise ValueError(
            "Detected disk is not brighter than space; inspect this image."
        )
    return disk, rho, mask


def apply_solar_disk_filter(input_path, output_path, config=None, **options):
    """File API for the disk stage; accepts mask_path/disk_path/reference_path."""
    from .cli import apply_stage
    return apply_stage("disk", input_path, output_path, config=config, **options)


if __name__ == "__main__":
    from .cli import main
    main(["disk", *__import__("sys").argv[1:]])
