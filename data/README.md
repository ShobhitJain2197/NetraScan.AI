# NetraScan.AI Dataset

This directory contains dataset metadata and split manifests used for **NetraScan.AI**.

Raw medical images are **not stored in this repository**. The original datasets can be obtained from their respective sources below.

## Dataset Sources

The project currently uses data collected from the following datasets:

### 1. Cataract v01

A cataract image classification dataset containing three classes:

- Normal
- Immature Cataract
- Mature Cataract

**Source:**  
https://universe.roboflow.com/cataract/cataract-v01

**Version used:** v7

---

### 2. Dataset Katarak

A cataract classification dataset containing:

- Normal
- Immature Cataract
- Mature Cataract

**Source:**  
https://www.kaggle.com/datasets/rifdana/dataset-katarak-sinilis

---

### 3. SLID — Slit-Lamp Image Dataset

SLID contains annotated anterior-eye slit-lamp images with anatomical and lesion-level annotations.

It includes annotations for:

- Cataract
- Intraocular Lens
- Lens Dislocation
- Cornea
- Pupil
- Conjunctiva
- Other anterior-eye conditions

**Dataset Repository:**  
https://github.com/xumingyu-hub/SLID

**Research Paper:**  
https://doi.org/10.3389/fdgth.2025.1716501

---

## Current Classification Classes

The current NetraScan.AI classification pipeline uses three classes:

- Normal
- Immature Cataract
- Mature Cataract

SLID images are currently retained separately for additional analysis and future extensions rather than being directly merged into the primary three-class training dataset.

## Dataset Engineering Pipeline

Before model training, the datasets undergo:

- Dataset integrity verification
- Metadata generation
- SHA-256 exact duplicate detection
- Perceptual hashing (pHash)
- Cross-dataset duplicate analysis
- Near-duplicate analysis
- Visual-family grouping using Union-Find
- Class distribution analysis
- Leakage-aware train/validation/test splitting

The final three-class dataset contains **3,542 images** divided into:

| Split | Images |
|---|---:|
| Train | 2,487 |
| Validation | 536 |
| Test | 519 |

Related images belonging to the same detected visual family are kept within the same dataset partition to reduce train-validation-test leakage.

## Important Note

The dataset split is **visual-family-aware**, not guaranteed patient-level, because sufficient patient identifiers are not available across all source datasets.

The same dataset manifests will be used for **ViT, Swin Transformer, MobileViT, and the CNN baseline** to ensure a controlled architecture comparison.
