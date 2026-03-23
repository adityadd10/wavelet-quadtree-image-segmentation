"""Quadtree segmentation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QuadRegion:
    """Represents a rectangular quadtree region."""

    x: int
    y: int
    width: int
    height: int


def _split_region(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    std_threshold: float,
    min_size: int,
    regions: list[QuadRegion],
    depth: int = 0,
    max_depth: int = 6,
) -> None:
    region = image[y : y + height, x : x + width]

    if region.size == 0:
        return

    
    if depth >= max_depth:
        regions.append(QuadRegion(x=x, y=y, width=width, height=height))
        return

    
    if width <= min_size or height <= min_size or np.std(region) < std_threshold:
        regions.append(QuadRegion(x=x, y=y, width=width, height=height))
        return

    half_width = width // 2
    half_height = height // 2

    if half_width == 0 or half_height == 0:
        regions.append(QuadRegion(x=x, y=y, width=width, height=height))
        return

    # recursive split
    _split_region(image, x, y, half_width, half_height,
                  std_threshold, min_size, regions, depth+1, max_depth)

    _split_region(image, x + half_width, y, width - half_width, half_height,
                  std_threshold, min_size, regions, depth+1, max_depth)

    _split_region(image, x, y + half_height, half_width, height - half_height,
                  std_threshold, min_size, regions, depth+1, max_depth)

    _split_region(image, x + half_width, y + half_height,
                  width - half_width, height - half_height,
                  std_threshold, min_size, regions, depth+1, max_depth)

def merge_regions(label_map, image, merge_threshold=0.05, max_iter=5):
    new_labels = label_map.copy()

    for _ in range(max_iter):  # multiple passes
        unique_labels = np.unique(new_labels)

        means = {l: np.mean(image[new_labels == l]) for l in unique_labels}

        changed = False

        h, w = new_labels.shape

        for y in range(h - 1):
            for x in range(w - 1):
                current = new_labels[y, x]

                # right neighbor
                right = new_labels[y, x + 1]
                if current != right:
                    if abs(means[current] - means[right]) < merge_threshold:
                        new_labels[new_labels == right] = current
                        changed = True

                # bottom neighbor
                down = new_labels[y + 1, x]
                if current != down:
                    if abs(means[current] - means[down]) < merge_threshold:
                        new_labels[new_labels == down] = current
                        changed = True

        if not changed:
            break

    return new_labels

def _build_label_map(shape: tuple[int, int], regions: list[QuadRegion]) -> np.ndarray:
    """Assign one integer label to each final quadtree region."""
    label_map = np.zeros(shape, dtype=np.int32)
    for label, region in enumerate(regions):
        label_map[region.y : region.y + region.height, region.x : region.x + region.width] = label
    return label_map

def relabel_map(label_map):
    unique = np.unique(label_map)
    new_map = np.zeros_like(label_map)
    
    for i, val in enumerate(unique):
        new_map[label_map == val] = i
    
    return new_map

def segment_image(
    image: np.ndarray,
    std_threshold: float = 0.12,   
    min_size: int = 64,           
) -> tuple[np.ndarray, np.ndarray]:

    if image.ndim != 2:
        raise ValueError("Quadtree segmentation expects a 2D grayscale image.")

    height, width = image.shape
    regions: list[QuadRegion] = []

    _split_region(
        image=image,
        x=0,
        y=0,
        width=width,
        height=height,
        std_threshold=std_threshold,
        min_size=min_size,
        regions=regions,
        depth=0,
        max_depth=6,   
    )

    base = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    visualization = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    for region in regions:
        cv2.rectangle(
            visualization,
            (region.x, region.y),
            (region.x + region.width - 1, region.y + region.height - 1),
            (0, 255, 0),
            1,
        )

    label_map = _build_label_map(image.shape, regions)

    # improved merging
    label_map = merge_regions(label_map, image, merge_threshold=0.08, max_iter=5)

    label_map = relabel_map(label_map)

    return visualization, label_map
