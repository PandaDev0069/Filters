from pathlib import Path
import cv2

from bilateral_filter import apply_bilateral_filter
from median_filter import apply_median_filter
from clahe_filter import apply_clahe_filter

# Test the filters with a sample image
project_root = Path(__file__).resolve().parent.parent
input_image = (
    project_root
    / "MAGFilo_1.0_Kaggle_2026"
    / "train"
    / "train_images"
    / "20110119082634Lh.jpeg"
)

output_dir = Path("Filters/output")
output_dir.mkdir(parents=True, exist_ok=True)

# Case: 1: Apply Median filter and then Bilateral filter
output_dir = Path(output_dir / "median_filter_experiments")
base_image = cv2.imread(str(input_image))
if base_image is None:
    raise ValueError(f"Failed to read the image from {input_image}.")
cv2.imwrite(str(output_dir / "base_image.jpeg"), base_image)
apply_median_filter(
    str(input_image),
    str(output_dir / "median_filtered_image.jpeg"),
    kernel_size=3,
)
apply_bilateral_filter(
    str(output_dir / "median_filtered_image.jpeg"),
    str(output_dir / "median_bilateral_filtered_image.jpeg"),
    diameter=15,
    sigma_color=15,
    sigma_space=3,
)

# Case: 2: Apply CLAHE filter and then Bilateral filter
output_dir = Path(output_dir / "clahe_filter_experiments")
apply_clahe_filter(
    str(input_image),
    str(output_dir / "clahe_filtered_image.jpeg"),
    clip_limit=2.0,
    tile_grid_size=(4, 4),
)
apply_bilateral_filter(
    str(output_dir / "clahe_filtered_image.jpeg"),
    str(output_dir / "clahe_bilateral_filtered_image.jpeg"),
    diameter=15,
    sigma_color=15,
    sigma_space=3,
)

# Case: 3: Apply CLAHE filter and then Median filter
output_dir = Path(output_dir / "clahe_median_filter_experiments")
apply_clahe_filter(
    str(input_image),
    str(output_dir / "clahe_filtered_image.jpeg"),
    clip_limit=2.0,
    tile_grid_size=(4, 4),
)
apply_median_filter(
    str(output_dir / "clahe_filtered_image.jpeg"),
    str(output_dir / "clahe_median_filtered_image.jpeg"),
    kernel_size=3,
)
