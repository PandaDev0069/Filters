# Independent solar filament preprocessing filters

The processing and evaluation algorithms from
[`../Final/solar-filament-pre-processing.ipynb`](../Final/solar-filament-pre-processing.ipynb)
are available as ordinary Python modules. Each filter accepts arrays in memory;
the command line and `apply_*` helpers add file input/output. Importing a module
does not execute the notebook, discover a dataset, run experiments, or create files.

Run the commands below from the project root, `D:\filament-segmentation-2026`.
Use the existing project environment (`.venv\Scripts\python.exe`) in place of
`python` if your terminal has not activated it. Dependencies are listed in
[`requirements.txt`](requirements.txt).

## Notebook section map

| Section | Reusable file | Main entry points |
|---|---|---|
| 1. Settings | [`config.py`](config.py), [`image_io.py`](image_io.py) | `get_config`, `load_grayscale` |
| 2. Discover files and freeze split | [`dataset.py`](dataset.py) | `discover_records`, `freeze_manifest` |
| 3. Disk detection and masking | [`solar_disk_filter.py`](solar_disk_filter.py) | `detect_solar_disk`, `apply_solar_disk_filter` |
| 4. Median filtering | [`median_filter.py`](median_filter.py) | `median_filter` |
| 4. Limb-darkening correction | [`radial_correction.py`](radial_correction.py) | `radial_correction`, `apply_radial_correction` |
| 4. Polynomial background correction | [`polynomial_correction.py`](polynomial_correction.py) | `polynomial_correction`, `apply_polynomial_correction` |
| 4. CLAHE | [`clahe_filter.py`](clahe_filter.py) | `prepare_fixed_clahe_input`, `clahe_filter` |
| 4. Percentile normalization | [`normalization.py`](normalization.py) | `robust_unit`, `to_uint8`, `normalized_enhanced_from_clahe` |
| 5. Signed Laplacian | [`laplacian_filter.py`](laplacian_filter.py) | `signed_laplacian`, `make_neighbor_laplacian`, `effective_signed_kernel` |
| 5. Adaptive threshold | [`adaptive_threshold.py`](adaptive_threshold.py) | `candidate_region`, `select_candidates`, `adaptive_candidates` |
| 5. Connected component filtering | [`component_filter.py`](component_filter.py) | `filter_components` |
| 5. Optional closing | [`morphological_closing.py`](morphological_closing.py) | `conservative_closing` |
| 6. Stage inspection | [`previews.py`](previews.py), [`dataset.py`](dataset.py) | `show_stages`, `select_preview_records` |
| 7. COCO comparison | [`candidate_evaluation.py`](candidate_evaluation.py) | `decode_annotation`, `evaluate_candidates`, `candidate_confusion_image` |
| 8. Export/cache | [`export.py`](export.py) | `save_image_result`, `save_processed`, `cache_record`, `export_dataset` |
| 9. Preview run | [`experiments.py`](experiments.py) | `run_preview` |
| 10. Annotation size analysis | [`annotation_size_filter.py`](annotation_size_filter.py) | `fit_annotation_size_bounds` |
| 11. Apply size bounds | [`annotation_size_filter.py`](annotation_size_filter.py) | `filter_candidate_size_bounds`, `apply_annotation_size_filter` |
| 12. Individual annotators | [`candidate_evaluation.py`](candidate_evaluation.py), [`dataset.py`](dataset.py) | `evaluate_annotators`, `scoped_mask_metrics`, `select_annotator_cohorts` |
| 13. Whole-component edge rejection | [`radius_edge_filter.py`](radius_edge_filter.py) | `radius_edge_geometry`, `reject_radius_edge_components` |
| 14. Laplacian experiments | [`experiments.py`](experiments.py) | `laplacian_variants`, `run_sweep` |
| 15. Final normalization experiments | [`experiments.py`](experiments.py) | `normalization_variants`, `run_sweep` |
| 16. CLAHE experiments | [`experiments.py`](experiments.py) | `clahe_variants`, `run_sweep` |
| 17. Bilateral settings | [`bilateral_filter.py`](bilateral_filter.py), [`experiments.py`](experiments.py) | `bilateral_filter`, `bilateral_variants` |
| 18. Bilateral locations | [`experiments.py`](experiments.py), [`pipeline.py`](pipeline.py) | `bilateral_location_variants`, `preprocess_image` |
| 19. Conclusion/scorecard | [`reporting.py`](reporting.py) | `aggregate_trials`, `select_best`, `summarize_saved_sweep` |
| 20. Filament instances | [`instance_segmentation.py`](instance_segmentation.py) | `candidates_to_instances`, `evaluate_instances`, `export_instances` |

The notebook's galleries are represented by plotting and reporting helpers;
the long display cells and previously saved experimental results are not run
automatically. [`pipeline.py`](pipeline.py) composes the individual filters.

## Presets and parameter changes

```python
from Filters.config import get_config

baseline = get_config()               # Original CONFIG in notebook sections 1–9
selected = get_config("recommended")  # Retained settings from section 19

custom = get_config("recommended", median_kernel=5, clahe_clip_limit=1.5)
```

Each call returns an independent dictionary. Overrides of nested settings replace
that whole setting, so copy and update the nested dictionary when changing one field.

| Setting | Baseline | Recommended |
|---|---|---|
| Median | 3×3 | 3×3 |
| Radial correction | 128 radial bins; gain capped to 0.5–2.5 | Same |
| Polynomial correction | Degree 3; three fitting iterations | Same |
| Pre-CLAHE normalization | 1st–99.5th percentiles | Same |
| CLAHE | Clip 2, grid 8×8 | Clip 1, grid 4×4 |
| Final normalization | 1st–99.5th percentiles | 5th–99.5th percentiles |
| Laplacian | 3×3 eight-neighbor stencil, gain 1 | Same |
| Candidate threshold | 20 intervals, upper edge; immediate limb exclusion | Same |
| Closing | Off | Off |
| Annotation size/span rules | Off | Area 209–35,251 px²; maximum span 850 px |
| Whole-component edge rule | Off | Width `max(8, 0.02 * R)` |
| Bilateral | Off | Off |

The recommended preset is a snapshot of the notebook's selected training-preview
configuration. Its size bounds were fitted on this dataset's training annotations;
use `fit_annotation_size_bounds()` to refit them for another dataset. The modules
do not need the notebook or its cached configuration file at runtime.

## Run a single filter from the command line

```powershell
python -m Filters --help

# Median only, with an adjustable kernel
python -m Filters median input.jpeg median.png --set median_kernel=5

# Detect disk; also writes disk.png.json containing center and radius
python -m Filters disk input.jpeg disk.png

# Median then radial correction, using the same disk geometry
python -m Filters median input.jpeg median.png --mask disk.png
python -m Filters radial median.png radial.npy --disk disk.png.json

# Polynomial correction consumes the exact radial float array
python -m Filters polynomial radial.npy corrected.npy --disk disk.png.json

# Prepare the CLAHE input, then apply CLAHE and final normalization
python -m Filters prepare-clahe corrected.npy clahe-input.png --mask disk.png
python -m Filters clahe clahe-input.png clahe.png --mask disk.png --preset recommended
python -m Filters normalize clahe.png enhanced.png --mask disk.png --preset recommended

# Explore signed responses or form raw binary candidates
python -m Filters laplacian enhanced.png laplacian.npy --mask disk.png
python -m Filters threshold enhanced.png raw.png --mask disk.png

# Candidate postfilters, each independently callable
python -m Filters components raw.png components.png --mask disk.png
python -m Filters closing components.png closed.png --mask disk.png
python -m Filters size closed.png sized.png --preset recommended
python -m Filters radius sized.png candidates.png --disk disk.png.json --preset recommended
python -m Filters instances candidates.png labels.npy
```

New numerical modules also have individual module commands, for example:

```powershell
python -m Filters.radial_correction median.png radial.npy --disk disk.png.json
python -m Filters.laplacian_filter enhanced.png laplacian.npy --mask disk.png
```

Most new modules also expose a file helper. For example:

```python
from Filters.radial_correction import apply_radial_correction

apply_radial_correction("median.png", "radial.npy", disk_path="disk.png.json")
```

`--config experiment.json` accepts a settings object or a notebook export with a
`parameters` field. `--set KEY=JSON` overrides individual settings. For non-2048
images pass `--set strict_shape=false` or configure the expected dimensions.

The single-stage commands apply only the named transform. Use the preceding stage's
output as input. Supply `--mask` for the notebook's on-disk statistics and masking;
otherwise normalization/CLAHE/thresholding use the full frame. Radial, polynomial,
and radius stages accept `--disk` or an original `--reference` image to share geometry.

`laplacian` saves the signed response only. `threshold` computes the response and
scores the intensity intervals. To tune thresholds using an existing response,
call `select_candidates()` directly.

## Combine independent functions in Python

```python
from Filters.config import get_config
from Filters.image_io import load_grayscale
from Filters.solar_disk_filter import detect_solar_disk
from Filters.median_filter import median_filter
from Filters.radial_correction import radial_correction
from Filters.polynomial_correction import polynomial_correction
from Filters.clahe_filter import prepare_fixed_clahe_input, clahe_filter
from Filters.normalization import normalized_enhanced_from_clahe
from Filters.adaptive_threshold import adaptive_candidates

cfg = get_config("recommended")
image = load_grayscale("input.jpeg", cfg)
disk, rho, mask = detect_solar_disk(image, cfg)
median = median_filter(image, kernel_size=3, disk_mask=mask)
radial, centers, profile = radial_correction(median, rho, mask, cfg)
corrected, background, coefficients = polynomial_correction(radial, mask, disk, cfg)
clahe_input, pre_stats = prepare_fixed_clahe_input(corrected, mask, (1, 99.5))
clahe = clahe_filter(clahe_input, clip_limit=1, tile_grid_size=(4, 4), disk_mask=mask)
enhanced, post_stats = normalized_enhanced_from_clahe(clahe, mask, (5, 99.5))
laplacian, eligible, raw_candidates, diagnostics = adaptive_candidates(enhanced, mask, cfg)
```

`raw_candidates` precedes connected-component, closing, size, and radius filtering.
Use the combined pipeline when you want those downstream steps too:

```python
from Filters.pipeline import preprocess_image
from Filters.export import save_image_result

result = preprocess_image(image, cfg)
save_image_result(result, "Filters/output/my-experiment", cfg)

enhanced = result["enhanced"]
candidates = result["candidates"]
rgb = result["multi_channel"]  # Original, actual Laplacian input, candidates
```

Or run the entire selected pipeline:

```powershell
python -m Filters pipeline input.jpeg Filters/output/my-experiment --preset recommended
```

All numerical stages preserve image dimensions. Arrays remain in memory between
filters. PNG files store uint8 images and binary masks; `.npy`/`.npz` files preserve
floating corrections, signed responses, and integer instance labels exactly.
The commands reject saving those floats or labels as PNG. Sidecars record the
input path, configuration, and per-stage diagnostics. Pipeline exports include
all returned arrays and metadata.

## Optimize one filter independently

```python
from Filters.experiments import run_sweep, clahe_variants

report = run_sweep(
    {"sample": "input.jpeg"},
    clahe_variants(reference_clip=1, reference_grid=(4, 4)),
    preset="recommended",
    truths={"sample": "truth.png"},  # Optional binary foreground union
    output_dir="Filters/output/clahe-sweep",
)
print(report["best"])
```

Every variant starts from the original image and a fresh configuration. The disk
is detected once and frozen across variants. Set `freeze_geometry=False` when
tuning disk detection itself. Changing an upstream filter recomputes downstream
stages so candidate comparisons reflect the change.

Factories cover the notebook's Laplacian, normalization, CLAHE, bilateral settings,
and bilateral locations. You can supply your own variants for other parameters:

```python
variants = [
    {"id": "median3", "overrides": {"median_kernel": 3}},
    {"id": "median5", "overrides": {"median_kernel": 5}},
]
report = run_sweep({"sample": image}, variants, config=cfg)
```

Command-line experiments support one image at a time:

```powershell
python -m Filters.experiments clahe input.jpeg Filters/output/clahe-sweep --preset recommended
python -m Filters.experiments bilateral-location input.jpeg Filters/output/location-sweep --preset recommended --truth truth.png
```

Without truth masks, sweeps export outputs and candidate counts without selecting
a winner. With truth masks, they pool confusion counts and rank by pixel IoU,
then Dice, with original variant order breaking exact ties. Changing `CONFIG`
or presets is always explicit; sweeps do not promote their winner automatically.

CLI sweeps default to the recommended preset. Reproduce the notebook's sequential
optimization by passing the prior experiment's selected config into the next one;
independent sweeps are not a replay of its historical results. The bilateral
location factory uses diameter 5, sigmaColor 15, sigmaSpace 3 by default; pass your
chosen values if an earlier bilateral sweep selected different parameters.

To enable bilateral once in the combined pipeline:

```python
cfg["bilateral_filter"] = {
    "enabled": True,
    "position": "after_median",  # before_median, before_clahe, after_normalization
    "diameter": 5,
    "sigma_color": 15,
    "sigma_space": 3,
}
```

## Annotation size fitting, individual annotators, and instances

```python
from Filters.annotation_size_filter import fit_annotation_size_bounds
from Filters.candidate_evaluation import evaluate_annotators
from Filters.instance_segmentation import candidates_to_instances, evaluate_instances, export_instances

# coco and records come from your annotation JSON and frozen split manifest.
rule = fit_annotation_size_bounds(coco, records)
cfg["annotation_area_filter"] = rule

# Keep separate COCO image IDs as separate annotators.
rows, truths = evaluate_annotators(candidates, annotations_by_image, image_ids, disk_mask=mask)

labels, stats, count = candidates_to_instances(candidates)
export_instances(labels, "Filters/output/instances", image_name="sample")

# gt_masks contains original individual instances for ONE annotator.
scores, iou_matrix, matches, truth_union = evaluate_instances(labels, gt_masks)
```

Instance labeling preserves the binary foreground exactly; it does not repair
splits or merges. PQ is filament-only and class-agnostic, uses strict IoU > 0.5,
and preserves separate, potentially overlapping ground-truth instances. Evaluate
each annotator independently. Component PQ from section 7 remains a separate
candidate diagnostic.

## Existing scripts

The original file functions and demonstrations in `median_filter.py`,
`bilateral_filter.py`, `clahe_filter.py`, and `main.py` remain available.
`apply_clahe_filter()` retains its LAB-lightness behavior for color images;
the new array function `clahe_filter()` and the unified command use the notebook's
direct grayscale CLAHE. Use the unified CLI for lossless stage chaining.

## Verification

```powershell
python -m pytest Filters/tests/test_modular_pipeline.py -q
```

The tests compare baseline output and all bilateral insertion locations directly
with the notebook's definition cells. They also verify lossless file chaining,
training-only size fitting, independent parameter trials, instance matching,
configuration isolation, and command-line use. Runtime modules never load the
notebook; only parity tests read it as their reference.
