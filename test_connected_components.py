import cv2
import numpy as np


def clean_filaments(image: np.ndarray, threshold: int = 50, min_pixel_area: int = 20) -> np.ndarray:
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    clean = np.zeros_like(binary)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_pixel_area:
            clean[labels == label] = 255

    return clean


def test_connected_components_remove_small_dark_blobs_and_keep_filaments(tmp_path):
    image = np.full((40, 50), 180, dtype=np.uint8)
    image[5:10, 5:13] = 20
    image[20:22, 20:22] = 20
    image[30:33, 35:38] = 20
    image[10:15, 30:35] = 240

    input_path = tmp_path / "solar_filament.png"
    output_path = tmp_path / "clean_solar_data.png"
    assert cv2.imwrite(str(input_path), image)

    loaded = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    assert loaded is not None
    clean = clean_filaments(loaded, min_pixel_area=20)
    assert cv2.imwrite(str(output_path), clean)

    saved = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    assert saved is not None
    expected = np.zeros_like(image)
    expected[5:10, 5:13] = 255

    np.testing.assert_array_equal(saved, expected)
