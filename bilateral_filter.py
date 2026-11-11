import os
import cv2
from pathlib import Path

def apply_bilateral_filter(input_path, output_path, diameter=9, sigma_color=15, sigma_space=3):
    """
    Applies a bilateral filter to an image and saves the result.

    :param input_path: Path to the input image file.
    :param output_path: Path to save the filtered image.
    :param diameter: Diameter of each pixel neighborhood used during filtering.
    :param sigma_color: Filter sigma in the color space. A larger value means that farther colors within the pixel neighborhood will be mixed together.
    :param sigma_space: Filter sigma in the coordinate space. A larger value means that farther pixels will influence each other as long as their colors are close enough.
    """
    # Read the input image
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The input file {input_path} does not exist.")

    image = cv2.imread(input_path)

    if image is None:
        raise ValueError(
            f"Failed to read the image from {input_path}. Ensure it's a valid image file."
        )

    # Apply the bilateral filter
    filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

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
    output_dir = Path(__file__).resolve().parent / "output/bilateral_filter_experiments"
    input_image_1 = cv2.imread(str(input_image))
    if input_image_1 is None:
        raise ValueError(f"Failed to read the image from {input_image}.")
    cv2.imwrite(str(output_dir / "a_image.jpeg"), input_image_1)
    
    # Loop through different parameter settings for the bilateral filter
    for diameter in [5, 9, 15]:
        for sigma_color in [15, 35, 75]:
            for sigma_space in [3, 7, 15]:
                output_image = (
                    output_dir
                    / f"bilateral_d{diameter}_sc{sigma_color}_ss{sigma_space}.jpeg"
                )
                apply_bilateral_filter(
                    str(input_image),
                    str(output_image),
                    diameter=diameter,
                    sigma_color=sigma_color,
                    sigma_space=sigma_space,
                )