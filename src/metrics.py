"""Quantitative metrics for evaluating segmentation outputs."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.metrics import silhouette_score


def _intra_region_variance(image: np.ndarray, labels: np.ndarray) -> float:
    """Average variance of pixel intensities inside each segment."""
    variances: list[float] = []
    for label in np.unique(labels):
        region_pixels = image[labels == label]
        if region_pixels.size == 0:
            continue
        variances.append(float(np.var(region_pixels)))
    return float(np.mean(variances)) if variances else 0.0


def _edge_map_from_labels(labels: np.ndarray) -> np.ndarray:
    """Build a binary edge map from segmentation boundaries."""
    boundaries = np.zeros(labels.shape, dtype=np.uint8)
    boundaries[1:, :] |= (labels[1:, :] != labels[:-1, :]).astype(np.uint8)
    boundaries[:, 1:] |= (labels[:, 1:] != labels[:, :-1]).astype(np.uint8)
    return boundaries


def _edge_preservation(image: np.ndarray, labels: np.ndarray) -> float:
    """
    Measure how well segmentation boundaries align with image edges.

    The score is the fraction of segmentation-boundary pixels that overlap
    with Canny edges extracted from the smoothed grayscale image.
    """
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image_edges = cv2.Canny(image_uint8, 80, 160) > 0
    segmentation_edges = _edge_map_from_labels(labels) > 0

    boundary_pixels = int(segmentation_edges.sum())
    if boundary_pixels == 0:
        return 0.0

    overlap = np.logical_and(image_edges, segmentation_edges).sum()
    return float(overlap / boundary_pixels)


def _safe_silhouette_score(image: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute a silhouette score from grayscale intensities and segment labels.

    Returns 0.0 if the segmentation is not suitable for silhouette scoring.
    """
    flat_labels = labels.reshape(-1)
    unique_labels = np.unique(flat_labels)

    if unique_labels.size < 2 or unique_labels.size >= flat_labels.size:
        return 0.0

    features = image.reshape(-1, 1).astype(np.float32)

    # Limit sample size on large images to keep the GUI responsive.
    sample_size = min(5000, features.shape[0])
    try:
        return float(
            silhouette_score(features, flat_labels, metric="euclidean", sample_size=sample_size)
        )
    except Exception:
        return 0.0


def evaluate_segmentation(image: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute the requested metrics for one segmentation output."""
    return {
        "Intra-region Variance": _intra_region_variance(image, labels),
        "Silhouette Score": _safe_silhouette_score(image, labels),
        "Edge Preservation": _edge_preservation(image, labels),
        "Number of Segments": float(np.unique(labels).size),
    }
