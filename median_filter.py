import os
import cv2
from pathlib import Path


def apply_median_filter(input_path, output_path, kernel_size=3):
    """
    Applies a median filter to an image and saves the result.

    :param input_path: Path to the input image file.
    :param output_path: Path to save the filtered image.
    :param kernel_size: Size of the kernel to be used for the median filter. Must be an odd integer.
    """
    # Read the input image
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The input file {input_path} does not exist.")

    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("Kernel size must be a positive odd integer.")

    image = cv2.imread(input_path)

    if image is None:
        raise ValueError(
            f"Failed to read the image from {input_path}. Ensure it's a valid image file."
        )

    # Apply the median filter
    filtered_image = cv2.medianBlur(image, kernel_size)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, filtered_image)
    print(f"Filtered image saved to {output_path}")


def run_staged_kernel_experiment(
    input_path,
    output_dir,
    larger_kernel_size=9,
    small_kernel_size=3,
    total_passes=5,
):
    """
    Applies one larger median-filter pass, followed by 3x3 passes.

    Each experiment gets its own folder. The first output uses the larger
    kernel, while the remaining outputs use the smaller kernel.
    """
    if total_passes < 2:
        raise ValueError("Total passes must be at least 2.")

    experiment_dir = (
        Path(output_dir) / f"{larger_kernel_size}K_then_{small_kernel_size}K"
    )
    current_input = Path(input_path)

    for filter_pass in range(1, total_passes + 1):
        kernel_size = larger_kernel_size if filter_pass == 1 else small_kernel_size
        output_image = experiment_dir / f"filtered_image_{filter_pass}times.jpeg"
        apply_median_filter(
            str(current_input),
            str(output_image),
            kernel_size=kernel_size,
        )
        current_input = output_image


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    input_image = (
        project_root
        / "MAGFilo_1.0_Kaggle_2026"
        / "train"
        / "train_images"
        / "20110119082634Lh.jpeg"
    )
    output_dir = Path(__file__).resolve().parent / "output/median_filter_experiments"
    # for larger_kernel_size in (5, 7, 9, 11, 13):
    #     run_staged_kernel_experiment(
    #         str(input_image),
    #         output_dir,
    #         larger_kernel_size=larger_kernel_size,
    #         small_kernel_size=3,
    #         total_passes=5,
    #     )

    # for kernel_size in range(3, 100, 2):
    #     current_input = input_image
    #     output_image = output_dir / f"{kernel_size}K_filtered_image.jpeg"
    #     apply_median_filter(
    #         str(current_input),
    #         str(output_image),
    #         kernel_size=kernel_size,
    #     )

    for kernel_size in range(3, 10, 2):
        current_input = input_image
        kernel_output_dir = output_dir / f"{kernel_size}K"
        input_image_1 = cv2.imread(str(input_image))
        if input_image_1 is None:
                raise ValueError(f"Failed to read the image from {input_image}.")
        cv2.imwrite(str(kernel_output_dir / "a_image.jpeg"), input_image_1)

        for filter_pass in range(1, 11):
            output_image = kernel_output_dir / f"filtered_image_{filter_pass}times.jpeg"
            apply_median_filter(
                str(current_input),
                str(output_image),
                kernel_size=kernel_size,
            )
            current_input = output_image
