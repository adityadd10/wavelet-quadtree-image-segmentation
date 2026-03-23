## Project Description

This project presents a complete satellite image processing pipeline that integrates signal processing and image segmentation techniques. The workflow begins with wavelet-based smoothing using Daubechies wavelets (db4/db6) to reduce noise while preserving important structural details.

The processed image is then segmented using a quadtree-based region segmentation approach, which recursively divides the image into homogeneous regions based on intensity variance. To improve segmentation quality and reduce over-segmentation, region merging and depth constraints are incorporated.

For comparison, K-means clustering is applied as a pixel-based segmentation method. The results of both approaches are evaluated using quantitative metrics such as intra-region variance, edge preservation, and silhouette score (for K-means).

A graphical user interface (GUI) is developed to allow users to upload satellite images, run the processing pipeline, and visualize the results interactively.

This project demonstrates the trade-off between spatially coherent segmentation (quadtree) and intensity-based clustering (K-means), highlighting their strengths and limitations in remote sensing applications.
