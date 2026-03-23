"""Tkinter GUI application for satellite image processing."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.metrics import evaluate_segmentation
from src.quadtree import segment_image as quadtree_segment
from src.segmentation import kmeans_segment
from src.wavelet import smooth_image


DISPLAY_SIZE = (320, 240)


def resize_for_display(image: np.ndarray, size: tuple[int, int] = DISPLAY_SIZE) -> Image.Image:
    """Resize a numpy image array to a display-friendly PIL image."""
    pil_image = Image.fromarray(image)
    pil_image.thumbnail(size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", size, color=(245, 247, 250))
    offset_x = (size[0] - pil_image.width) // 2
    offset_y = (size[1] - pil_image.height) // 2
    canvas.paste(pil_image, (offset_x, offset_y))
    return canvas


def normalize_image(gray_image: np.ndarray) -> np.ndarray:
    """Normalize a grayscale image to the range [0, 1]."""
    gray_float = gray_image.astype(np.float32)
    return cv2.normalize(gray_float, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)


class SatelliteImageApp:
    """Simple desktop application for the image processing workflow."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Satellite Image Processing")
        self.root.geometry("1400x760")
        self.root.minsize(1100, 700)

        self.image_path: Path | None = None
        self.source_bgr: np.ndarray | None = None
        self.photo_cache: dict[str, ImageTk.PhotoImage] = {}
        self.metrics_text: tk.Text | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        main_frame = ttk.Frame(self.root, padding=16)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 16))

        ttk.Label(
            control_frame,
            text="Satellite Image Processing Pipeline",
            font=("Segoe UI", 16, "bold"),
        ).pack(side="left")

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side="right")

        ttk.Button(button_frame, text="Upload Image", command=self.upload_image).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(button_frame, text="Process Image", command=self.process_image).pack(side="left")

        self.status_var = tk.StringVar(value="Choose a satellite image to begin.")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=1, column=0, sticky="ew", pady=(0, 12))

        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=2, column=0, sticky="nsew")

        for column in range(2):
            results_frame.columnconfigure(column, weight=1)
        for row in range(2):
            results_frame.rowconfigure(row, weight=1)

        self.image_labels: dict[str, ttk.Label] = {}
        cards = [
            ("original", "Original Image"),
            ("wavelet", "Wavelet Smoothed Image"),
            ("quadtree", "Quadtree Segmented Image"),
            ("kmeans", "K-Means Segmentation"),
        ]

        for index, (key, title) in enumerate(cards):
            row = index // 2
            column = index % 2

            card = ttk.LabelFrame(results_frame, text=title, padding=12)
            card.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)
            card.columnconfigure(0, weight=1)
            card.rowconfigure(0, weight=1)

            label = ttk.Label(card, anchor="center")
            label.grid(row=0, column=0, sticky="nsew")
            self.image_labels[key] = label

            self._set_placeholder(key, title)

        metrics_frame = ttk.LabelFrame(main_frame, text="Quantitative Metrics", padding=12)
        metrics_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        metrics_frame.columnconfigure(0, weight=1)

        self.metrics_text = tk.Text(
            metrics_frame,
            height=7,
            wrap="none",
            font=("Consolas", 10),
            relief="flat",
            borderwidth=0,
            background="#f7f7f7",
        )
        self.metrics_text.grid(row=0, column=0, sticky="ew")
        self._set_metrics_text("Metrics will appear here after processing the segmentation methods.")

    def _set_placeholder(self, key: str, title: str) -> None:
        """Show placeholder text before an image is available."""
        self.image_labels[key].configure(text=f"{title}\n\nNo image available", image="")

    def _show_image(self, key: str, image: np.ndarray) -> None:
        """Convert numpy array to Tk image and display it."""
        preview = resize_for_display(image)
        tk_image = ImageTk.PhotoImage(preview)
        self.photo_cache[key] = tk_image
        self.image_labels[key].configure(image=tk_image, text="")

    def upload_image(self) -> None:
        """Open a file dialog and load the selected image."""
        file_path = filedialog.askopenfilename(
            title="Select a satellite image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )

        if not file_path:
            return

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Invalid Image", "Unable to open the selected file as an image.")
            return

        self.image_path = Path(file_path)
        self.source_bgr = image

        original_rgb = cv2.cvtColor(self.source_bgr, cv2.COLOR_BGR2RGB)
        self._show_image("original", original_rgb)
        self._set_placeholder("wavelet", "Wavelet Smoothed Image")
        self._set_placeholder("quadtree", "Quadtree Segmented Image")
        self._set_placeholder("kmeans", "K-Means Segmentation")
        self._set_metrics_text("Metrics will appear here after processing the segmentation methods.")
        self.status_var.set(f"Loaded image: {self.image_path.name}")

    def process_image(self) -> None:
        """Run the full processing pipeline and display every stage."""
        if self.source_bgr is None:
            messagebox.showwarning("No Image", "Please upload an image before processing.")
            return

        try:
            gray = cv2.cvtColor(self.source_bgr, cv2.COLOR_BGR2GRAY)
            normalized = normalize_image(gray)

            smoothed = smooth_image(normalized, wavelet_name="db4", level=2, threshold_scale=0.35)
            quadtree_visual, quadtree_labels = quadtree_segment(
                smoothed, std_threshold=0.08, min_size=16
            )
            kmeans_visual, kmeans_labels = kmeans_segment(smoothed, clusters=4)
            quadtree_metrics = evaluate_segmentation(smoothed, quadtree_labels)
            kmeans_metrics = evaluate_segmentation(smoothed, kmeans_labels)

            smoothed_rgb = cv2.cvtColor((smoothed * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            quadtree_rgb = cv2.cvtColor(quadtree_visual, cv2.COLOR_BGR2RGB)
            kmeans_rgb = cv2.cvtColor(kmeans_visual, cv2.COLOR_BGR2RGB)

            self._show_image("wavelet", smoothed_rgb)
            self._show_image("quadtree", quadtree_rgb)
            self._show_image("kmeans", kmeans_rgb)
            self._set_metrics_text(self._format_metrics(quadtree_metrics, kmeans_metrics))
            self.status_var.set("Processing completed successfully.")
        except Exception as exc:
            messagebox.showerror("Processing Error", f"An error occurred while processing the image:\n{exc}")

    def _set_metrics_text(self, text: str) -> None:
        """Safely update the metrics display area."""
        if self.metrics_text is None:
            return
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", text)
        self.metrics_text.configure(state="disabled")

    def _format_metrics(
        self,
        quadtree_metrics: dict[str, float],
        kmeans_metrics: dict[str, float],
    ) -> str:
        """Format segmentation metrics into a side-by-side text block."""
        lines = [f"{'Metric':<24} {'Quadtree':>12} {'K-Means':>12}"]
        for label in quadtree_metrics:
            quad_value = quadtree_metrics[label]
            kmeans_value = kmeans_metrics[label]
            if label == "Number of Segments":
                lines.append(f"{label:<24} {int(quad_value):>12} {int(kmeans_value):>12}")
            else:
                lines.append(f"{label:<24} {quad_value:>12.4f} {kmeans_value:>12.4f}")
        return "\n".join(lines)


def main() -> None:
    """Application entry point."""
    root = tk.Tk()
    ttk.Style().theme_use("clam")
    SatelliteImageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
