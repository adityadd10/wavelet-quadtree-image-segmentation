"""K-Means segmentation utilities."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans


def kmeans_segment(
    image: np.ndarray,
    clusters: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment a normalized grayscale image using K-Means clustering.

    Returns a BGR visualization and the integer label map.
    """
    if image.ndim != 2:
        raise ValueError("K-Means segmentation expects a 2D grayscale image.")

    pixels = image.reshape(-1, 1).astype(np.float32)
    model = KMeans(n_clusters=clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(pixels)
    centers = model.cluster_centers_.flatten()

    segmented = centers[labels].reshape(image.shape)
    segmented = np.clip(segmented, 0.0, 1.0)
    segmented_uint8 = (segmented * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(segmented_uint8, cv2.COLORMAP_VIRIDIS)
    label_map = labels.reshape(image.shape).astype(np.int32)
    return colored, label_map
