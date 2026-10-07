# NetraScan.AI Dataset

This directory contains dataset metadata and split manifests used for NetraScan.AI.

Raw medical images are not stored in this repository.

## Dataset Sources

The project currently uses data collected from:

- Cataract v01
- Dataset Katarak
- SLID (Slit-Lamp Image Dataset)

## Current Classification Classes

- Normal
- Immature Cataract
- Mature Cataract

The dataset pipeline includes duplicate detection, perceptual similarity analysis, visual-family grouping, and leakage-aware train/validation/test splitting.
