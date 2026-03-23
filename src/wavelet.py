"""Wavelet-based image smoothing utilities."""

from __future__ import annotations

import numpy as np
import pywt


def smooth_image(
    image: np.ndarray,
    wavelet_name: str = "db4",
    level: int = 2,
    threshold_scale: float = 0.35,
) -> np.ndarray:
    """
    Smooth a normalized grayscale image using wavelet thresholding.

    Parameters
    ----------
    image:
        2D normalized grayscale image with values in the range [0, 1].
    wavelet_name:
        Daubechies wavelet family member, typically ``db4`` or ``db6``.
    level:
        Decomposition level used for the wavelet transform.
    threshold_scale:
        Multiplier applied to the estimated universal threshold.

    Returns
    -------
    np.ndarray
        Smoothed image with values clipped to the range [0, 1].
    """
    if image.ndim != 2:
        raise ValueError("Wavelet smoothing expects a 2D grayscale image.")

    if wavelet_name not in {"db4", "db6"}:
        raise ValueError("wavelet_name must be 'db4' or 'db6'.")

    coeffs = pywt.wavedec2(image, wavelet=wavelet_name, level=level)
    approximation = coeffs[0]
    detail_coeffs = coeffs[1:]

    # Estimate a noise-aware threshold using the first level high-frequency band.
    reference_band = detail_coeffs[-1][0]
    sigma = np.median(np.abs(reference_band)) / 0.6745 if reference_band.size else 0.0
    threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(image.size + 1.0))

    filtered_details = []
    for horizontal, vertical, diagonal in detail_coeffs:
        filtered_details.append(
            (
                pywt.threshold(horizontal, threshold, mode="soft"),
                pywt.threshold(vertical, threshold, mode="soft"),
                pywt.threshold(diagonal, threshold, mode="soft"),
            )
        )

    reconstructed = pywt.waverec2([approximation, *filtered_details], wavelet=wavelet_name)
    reconstructed = reconstructed[: image.shape[0], : image.shape[1]]
    return np.clip(reconstructed, 0.0, 1.0)
