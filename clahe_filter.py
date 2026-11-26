import os
import cv2
from pathlib import Path

if __package__:
    from .bilateral_filter import apply_bilateral_filter
else:
    from bilateral_filter import apply_bilateral_filter


def prepare_fixed_clahe_input(corrected, disk_mask, percentiles=(1.0, 99.5)):
    """Scale the corrected float image and fill space with a neutral intensity."""
    import numpy as np
    if __package__:
        from .normalization import robust_unit, to_uint8
    else:
        from normalization import robust_unit, to_uint8

    unit, stats = robust_unit(corrected, disk_mask, percentiles)
    image = to_uint8(unit)
    image[~disk_mask] = int(np.median(image[disk_mask]))
    return image, stats


def clahe_filter(image, clip_limit=2.0, tile_grid_size=(8, 8), disk_mask=None):
    """Apply the notebook's grayscale CLAHE to a prepared uint8 array.

    The existing apply_clahe_filter() below keeps its LAB-color behavior.
    Use prepare_fixed_clahe_input() first for polynomial-corrected floats.
    """
    import math
    import numpy as np

    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Notebook CLAHE requires a two-dimensional uint8 image.")
    if not math.isfinite(clip_limit) or clip_limit <= 0:
        raise ValueError("CLAHE clip limit must be finite and positive.")
    grid = tuple(tile_grid_size)
    if (len(grid) != 2 or any(int(v) != v or v < 1 for v in grid)
            or grid[0] > image.shape[1] or grid[1] > image.shape[0]):
        raise ValueError("Tile counts must be positive integers within the image dimensions.")
    result = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=tuple(map(int, grid))
    ).apply(image)
    if disk_mask is not None:
        if disk_mask.shape != image.shape:
            raise ValueError("Disk mask must match the image dimensions.")
        result[~disk_mask] = 0
    return result


def apply_clahe_setting(clahe_input, disk_mask, variant):
    """Notebook-compatible wrapper for a clip_limit/tile_grid variant."""
    return clahe_filter(clahe_input, variant["clip_limit"], variant["tile_grid"], disk_mask)


def apply_clahe_filter(input_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image and saves the result.

    :param input_path: Path to the input image file.
    :param output_path: Path to save the filtered image.
    :param clip_limit: Threshold for contrast limiting. Higher values give more contrast.
    :param tile_grid_size: Size of the grid for the histogram equalization. It should be a tuple of two integers.
    """
    # Read the input image
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The input file {input_path} does not exist.")

    image = cv2.imread(input_path)

    if image is None:
        raise ValueError(
            f"Failed to read the image from {input_path}. Ensure it's a valid image file."
        )

    # Convert the image to grayscale if it is not already
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into its channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L-channel back with A and B channels
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    filtered_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, filtered_image)
    print(f"Filtered image saved to {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    input_image = (
        project_root
        / "MAGFilo_1.0_Kaggle_2026"
        / "train"
        / "train_images"
        / "20110119082634Lh.jpeg"
    )
    output_dir = Path(__file__).resolve().parent / "output/clahe_filter_experiments"
    output_image_path = output_dir / "clahe_filtered_image.jpeg"
    input_image_1 = cv2.imread(str(input_image))
    if input_image_1 is None:
        raise ValueError(f"Failed to read the image from {input_image}.")
    cv2.imwrite(str(output_dir / "a_image.jpeg"), input_image_1)
    if input_image_1 is None:
        raise ValueError(f"Failed to read the image from {input_image}.")
    cv2.imwrite(str(output_dir / "a_image.jpeg"), input_image_1)
    apply_clahe_filter(str(input_image), str(output_dir / "clahe_filtered_image.jpeg"))
    # apply_bilateral_filter(
    #     output_image_path,
    #     str(output_dir / "clahe_and_bilateral_image.jpeg"),
    #     diameter=15,
    #     sigma_color=15,
    #     sigma_space=3,
    # )
