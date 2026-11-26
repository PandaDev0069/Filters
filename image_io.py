"""Validated image input and lossless array output shared by the filters."""

from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from PIL import Image


def load_array(path):
    """Load exact .npy/.npz stages or an image; .npz must contain one array."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            if len(arrays.files) != 1:
                raise ValueError("Stage NPZ input must contain exactly one array.")
            return arrays[arrays.files[0]].copy()
    with Image.open(path) as image:
        return np.asarray(image).copy()


def load_mask(path, shape=None):
    """Accept binary 0/1 or 0/255 masks, rejecting accidental grayscale inputs."""
    array = load_array(path)
    if (array.ndim != 2 or not np.isin(array, [0, 1, 255]).all()
            or (shape is not None and array.shape != tuple(shape))):
        raise ValueError("Expected a binary mask with matching two-dimensional shape.")
    return array.astype(bool)


def save_array(path, array):
    """Use PNG for uint8/bool and NumPy files for floats or instance labels."""
    path = Path(path)
    array = np.asarray(array)
    if path.suffix.lower() in (".npy", ".npz"):
        write_numpy(path, array=array)
    elif path.suffix.lower() == ".png" and array.dtype in (np.dtype("uint8"), np.dtype("bool")):
        write_png(path, array.astype(np.uint8) * 255 if array.dtype == bool else array,
                  rgb=array.ndim == 3 and array.shape[-1] == 3)
    else:
        raise ValueError("Use .png for uint8/bool or .npy/.npz for exact float/label arrays.")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_grayscale(path, cfg, coco_shape=None):
    # Check the source dtype rather than silently truncating a 16-bit image to uint8.
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            raise ValueError(f"{path}: expected uint8 source, got {array.dtype}")
        gray = np.asarray(image.convert("L")).copy()
    if gray.ndim != 2 or not gray.size or not np.isfinite(gray).all():
        raise ValueError(f"{path}: expected a nonempty, finite 2-D image")
    if cfg["strict_shape"] and tuple(gray.shape) != tuple(cfg["expected_shape"]):
        raise ValueError(f'{path}: expected {cfg["expected_shape"]}, got {gray.shape}')
    if coco_shape and tuple(gray.shape) != tuple(coco_shape):
        raise ValueError(f"{path}: source shape disagrees with COCO metadata")
    return np.ascontiguousarray(gray)


def write_png(path, array, rgb=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_input = cv2.cvtColor(array, cv2.COLOR_RGB2BGR) if rgb else array
    success, buffer = cv2.imencode(
        ".png", encoded_input, [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    if not success:
        raise OSError(f"PNG encoding failed: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(buffer.tobytes())
    temporary.replace(path)


def write_numpy(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        if path.suffix == ".npy":
            np.save(stream, next(iter(arrays.values())), allow_pickle=False)
        else:
            np.savez_compressed(stream, **arrays)
    temporary.replace(path)
