# NetraScan.AI Dataset

This directory contains the dataset metadata, visual-family assignments
and split manifests used in the NetraScan.AI research project.

Raw medical images are not distributed with this repository.

## 1. Dataset Sources

### Cataract v01

- Version: v7
- Images included: 587
- Classes: Normal, Immature Cataract, Mature Cataract
- Source: https://universe.roboflow.com/cataract/cataract-v01

### Dataset Katarak

- Images included: 2,955
- Classes: Normal, Immature Cataract, Mature Cataract
- Source: https://www.kaggle.com/datasets/rifdana/dataset-katarak-sinilis

### SLID - Slit-Lamp Image Dataset

SLID provides annotated anterior-eye slit-lamp images and anatomical
and lesion-level annotations.

- Repository: https://github.com/xumingyu-hub/SLID
- Research paper: https://doi.org/10.3389/fdgth.2025.1716501

SLID was considered separately for additional analysis and future
extensions. It was not merged into the final V4 three-class
classification dataset.

## 2. Classification Categories

| Class index | Category |
|---:|---|
| 0 | Normal |
| 1 | Immature Cataract |
| 2 | Mature Cataract |

## 3. Dataset Engineering Pipeline

The dataset preparation workflow includes:

- Dataset integrity verification
- Metadata generation
- SHA-256 exact duplicate analysis
- Perceptual hashing (pHash)
- Cross-dataset duplicate analysis
- Near-duplicate analysis
- Visual-family grouping using Union-Find
- Class distribution analysis
- Leakage-aware dataset partitioning

## 4. Historical and Final Dataset Splits

Both historical and final V4 manifests are retained for
research traceability.

| Partition | Historical split | Final V4 split |
|---|---:|---:|
| Training | 2,487 | 2,454 |
| Validation | 536 | 517 |
| Testing | 519 | 571 |
| Total | 3,542 | 3,542 |

The final V4 split is the reference partition for the reported
ViT-B/16, Swin-Tiny and MobileViT-S evaluation results.

## 5. Final V4 Class Distribution

| Category | Training | Validation | Testing | Total |
|---|---:|---:|---:|---:|
| Normal | 799 | 178 | 192 | 1,169 |
| Immature Cataract | 837 | 164 | 195 | 1,196 |
| Mature Cataract | 818 | 175 | 184 | 1,177 |
| Total | 2,454 | 517 | 571 | 3,542 |

## 6. Final V4 Source Distribution

| Source | Training | Validation | Testing | Total |
|---|---:|---:|---:|---:|
| Cataract v01 | 386 | 75 | 126 | 587 |
| Dataset Katarak | 2,068 | 442 | 445 | 2,955 |
| Total | 2,454 | 517 | 571 | 3,542 |

## 7. Dataset Integrity Checks

| Integrity criterion | Observed result |
|---|---:|
| Total images | 3,542 |
| Detected visual families | 2,746 |
| Visual families crossing partitions | 0 |
| Cross-partition pHash pairs at distance <= 4 | 0 |
| Detected Roboflow lineages crossing partitions | 0 |
| Missing image paths during original verification | 0 |

These checks reduce detectable image-level leakage but do not
establish patient-level independence.

## 8. Metadata Organization

The historical manifests are located directly under `metadata/`.

The final V4 manifests and split configuration are located under
`metadata/v4/`.

Each image metadata record contains:

- Dataset source
- Original split
- Filename
- Classification label
- Relative image path
- SHA-256 hash
- Perceptual hash
- Visual family identifier
- Visual family size
- Assigned experimental split

## 9. Reproducing the Dataset Layout

Download the original datasets from their respective sources.

The relative image paths assume the following directory structure:

```
DATASET_ROOT/
    Cataract v01.v7i.folder/
        train/
            normal/
            immature/
            mature/
    Dataset Katarak/
        train/
            normal/
            immature/
            mature/
        test/
            normal/
            immature/
            mature/
```

Configure the dataset root in the relevant research notebook.

The notebooks use example Windows workspace paths, such as
`C:\NetraScanWorkspace\MY PROJECT E PAARVAI`, which must be
adapted to the local environment.

The historical CSV manifests have been normalized to relative paths
for portability. Their labels, hashes, visual-family assignments
and partition assignments are unchanged.

The final V4 metadata remains unchanged.

## 10. Research Limitations

The dataset partitioning is visual-family-aware rather than
guaranteed patient-level.

External clinical validation, patient-level independence and
clinical probability calibration have not been established.

Users must comply with the original dataset licensing and
redistribution requirements.

NetraScan.AI is an academic research prototype and is not intended
for independent clinical diagnosis.

---

**Created by Mr. SHOBHIT JAIN**

**Three Transformers. One Vision.**
