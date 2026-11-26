"""Independent baseline and selected notebook presets, plus local overrides."""

from copy import deepcopy

PIPELINE_VERSION = "solar-filament-preprocessing-v3"
ANNOTATION_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

BASELINE_CONFIG = {'expected_shape': [2048, 2048],
 'strict_shape': True,
 'hough_coarse_size': 512,
 'hough_fine_size': 1024,
 'hough_param1': 80.0,
 'hough_param2': 35.0,
 'disk_radius_fraction': [0.25, 0.49],
 'median_kernel': 3,
 'radial_bins': 128,
 'radial_sample_stride': 4,
 'radial_gain_limits': [0.5, 2.5],
 'polynomial_degree': 3,
 'polynomial_sample_stride': 16,
 'polynomial_clip_sigma': 3.0,
 'polynomial_iterations': 3,
 'clahe_clip_limit': 2.0,
 'clahe_grid': [8, 8],
 'normalization_percentiles': [1.0, 99.5],
 'threshold_intervals': 20,
 'threshold_sigma_span': 3.0,
 'threshold_edge': 'upper',
 'candidate_limb_margin_px': 2,
 'min_component_area_px': 16,
 'min_component_disk_fraction': 1e-05,
 'closing_kernel': 0,
 'save_intermediates': True,
 'save_float_stages': True}

RECOMMENDED_CONFIG = {'expected_shape': [2048, 2048],
 'strict_shape': True,
 'hough_coarse_size': 512,
 'hough_fine_size': 1024,
 'hough_param1': 80.0,
 'hough_param2': 35.0,
 'disk_radius_fraction': [0.25, 0.49],
 'median_kernel': 3,
 'radial_bins': 128,
 'radial_sample_stride': 4,
 'radial_gain_limits': [0.5, 2.5],
 'polynomial_degree': 3,
 'polynomial_sample_stride': 16,
 'polynomial_clip_sigma': 3.0,
 'polynomial_iterations': 3,
 'clahe_clip_limit': 1.0,
 'clahe_grid': [4, 4],
 'normalization_percentiles': [1.0, 99.5],
 'threshold_intervals': 20,
 'threshold_sigma_span': 3.0,
 'threshold_edge': 'upper',
 'candidate_limb_margin_px': 2,
 'min_component_area_px': 16,
 'min_component_disk_fraction': 1e-05,
 'closing_kernel': 0,
 'save_intermediates': True,
 'save_float_stages': True,
 'post_clahe_normalization_percentiles': [5.0, 99.5],
 'signed_laplacian_variant': {'id': 'baseline_8n',
                              'label': '3×3 eight-neighbor (baseline)',
                              'gain': 1.0,
                              'control': False,
                              'raw_kernel': [[1.0, 1.0, 1.0],
                                             [1.0, -8.0, 1.0],
                                             [1.0, 1.0, 1.0]],
                              'effective_kernel': [[1.0, 1.0, 1.0],
                                                   [1.0, -8.0, 1.0],
                                                   [1.0, 1.0, 1.0]]},
 'annotation_area_filter': {'fit_split': 'train',
                            'measurement': 'native-resolution rasterized '
                                           'annotation-instance area',
                            'candidate_connectivity': 8,
                            'quantile': 0.01,
                            'observed_minimum_area_px': 9,
                            'observed_maximum_area_px': 35251,
                            'minimum_candidate_area_px': 209,
                            'maximum_area_quantile': 1.0,
                            'maximum_span_quantile': 1.0,
                            'maximum_candidate_area_px': 35251,
                            'maximum_candidate_bbox_span_px': 850,
                            'comparison': 'minimum <= component area <= maximum; bbox '
                                          'longest side <= maximum span',
                            'measured_instances': 6347,
                            'empty_masks_excluded': 0,
                            'crowd_annotations_excluded': 0,
                            'training_instances_below_cutoff': 64,
                            'training_instances_above_area_maximum': 0,
                            'training_instances_above_span_maximum': 0,
                            'training_instance_area_retention': 0.9899164959823539,
                            'training_instance_size_retention': 0.9899164959823539,
                            'annotation_sha256': '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',
                            'split_manifest_sha256': '777bc6e9f0ee5f4fb790492a7d30cbb0859d89d06f14ca6cacec8a55ee0f7fae'},
 'radius_edge_filter': {'minimum_edge_width_px': 8.0,
                        'edge_width_radius_fraction': 0.02,
                        'connectivity': 8,
                        'removal_mode': 'whole_component_touching_or_outside_safe_radius'},
 'bilateral_filter': {'id': 'd5_c15_s3',
                      'label': 'd5, C15, S3',
                      'enabled': False,
                      'diameter': 5,
                      'sigma_color': 15.0,
                      'sigma_space': 3.0,
                      'position': 'none'},
 'disk_detection_scope': 'original image; shared geometry across positions'}

RADIUS_EDGE_CONFIG = {
    "minimum_edge_width_px": 8.0,
    "edge_width_radius_fraction": 0.02,
    "connectivity": 8,
    "removal_mode": "whole_component_touching_or_outside_safe_radius",
}


def get_config(preset="baseline", **overrides):
    """Return a fresh config; nested overrides replace the corresponding setting.

    Recommended size bounds were fitted on this dataset's training annotations.
    Refit them for a different dataset with fit_annotation_size_bounds().
    """
    presets = {"baseline": BASELINE_CONFIG, "recommended": RECOMMENDED_CONFIG}
    if preset not in presets:
        raise ValueError("preset must be baseline or recommended")
    config = deepcopy(presets[preset])
    unknown = set(overrides) - (set(BASELINE_CONFIG) | set(RECOMMENDED_CONFIG))
    if unknown:
        raise ValueError(f"Unknown configuration settings: {sorted(unknown)}")
    config.update(deepcopy(overrides))
    return config
