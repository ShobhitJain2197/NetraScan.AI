# NetraScan.AI

### Multi-Transformer Cataract Classification & Explainable AI

**An AI-powered academic research prototype for cataract classification using Vision Transformer (ViT), Swin Transformer, and MobileViT, with ensemble inference, Grad-CAM explainability, and automated report generation.**

**Created by Mr. SHOBHIT JAIN**

---

## 1. Project Overview

NetraScan.AI is a deep learning-based cataract classification system designed to analyze eye images and categorize them into three classes:

- **Normal**
- **Immature Cataract**
- **Mature Cataract**

The system integrates three transformer-based computer vision architectures and combines their predictions using majority voting.

An interactive Gradio interface enables users to upload eye images, inspect model confidence scores, visualize Grad-CAM activation maps, and generate structured classification reports.

The project is intended for academic research, experimentation, and demonstration—not clinical diagnosis.

## 2. Key Features

- **Multi-Transformer Architecture:** Integrates ViT-B/16, Swin-Tiny, and MobileViT-S.
- **Ensemble Classification:** Combines predictions from all three models using majority voting.
- **Explainable AI:** Generates Swin-Tiny Grad-CAM activation heatmaps and image overlays.
- **Confidence Visualization:** Displays per-model class probabilities and inference latency.
- **Interactive Web Interface:** Built using Gradio with image upload and webcam input.
- **Automated Reporting:** Generates downloadable Markdown classification reports.
- **Portable Deployment:** Supports execution from an external SSD using a bundled Python runtime.
- **GPU Acceleration:** Uses NVIDIA CUDA when available, with CPU fallback.

## 3. Model Architectures

| Architecture | Implementation | Input Resolution |
|---|---|---|
| Vision Transformer | ViT-B/16 | 224 × 224 |
| Swin Transformer | Swin-Tiny | 224 × 224 |
| Mobile Vision Transformer | MobileViT-S | 256 × 256 |

### Vision Transformer (ViT-B/16)

ViT processes images as sequences of patches using transformer-based self-attention. This enables the model to learn relationships between different regions of an eye image.

### Swin Transformer (Swin-Tiny)

Swin Transformer uses hierarchical feature extraction and shifted-window self-attention. Its intermediate representations are also used for Grad-CAM visualization in this project.

### MobileViT-S

MobileViT combines convolutional operations with transformer-based global representations, providing a relatively lightweight architecture for image classification.

## 4. Dataset and Preprocessing

The project uses a consolidated eye-image dataset assembled from multiple sources, including:

- Cataract v01
- Dataset Katarak
- Slit-lamp image resources used during dataset investigation

The final V4 classification dataset contains **3,542 images**.

| Split | Images |
|---|---:|
| Training | 2,454 |
| Validation | 517 |
| Testing | 571 |
| **Total** | **3,542** |

The dataset engineering pipeline includes duplicate detection using perceptual hashing, visual-family grouping, class organization, preprocessing, and split-integrity verification.

Family-aware partitioning was used to reduce the risk of closely related images appearing across training, validation, and test sets.

## 5. Experimental Results

The following results were obtained on the project's internal V4 test set of 571 images.

| Model | Test Accuracy |
|---|---:|
| ViT-B/16 | 97.37% |
| Swin-Tiny | **98.95%** |
| MobileViT-S | 98.25% |
| Majority-Voting Ensemble | 98.25% |

**Swin-Tiny achieved the highest individual test accuracy** in the evaluated configuration.

These results describe performance on the internal evaluation dataset. They do not establish clinical accuracy or generalization to all real-world eye images.

## 6. Ensemble Classification

NetraScan.AI performs independent inference using three transformer models.

Each model produces class probabilities for Normal, Immature Cataract, and Mature Cataract.

The final ensemble prediction is determined using majority voting.

The interface also displays the number of models agreeing with the ensemble result.

## 7. Explainable Artificial Intelligence

The project incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)** for visual explanation.

Grad-CAM is applied to an intermediate Swin-Tiny feature representation to generate:

1. An activation heatmap.
2. A heatmap overlay on the original image.

These visualizations help researchers investigate which image regions influence model predictions.

**Important:** Grad-CAM highlights model activation patterns. It does not independently verify that a prediction is medically correct.

## 8. Automated Classification Reports

The application generates structured Markdown reports containing:

- Predicted cataract category
- Individual model predictions
- Confidence scores
- Inference latency
- Ensemble agreement
- Model architecture information
- Explainability notes
- Research limitations and disclaimers

The current reporting implementation is **template-based and deterministic**. LLM-generated clinical-language reporting is a potential future extension.

## 9. Project Structure

```text
NetraScan_GitHub/
│
├── app.py
├── README.md
├── requirements.txt
├── START_NETRASCAN.bat
├── .gitignore
│
├── src/
│   ├── inference.py
│   ├── explainability.py
│   └── report_generator.py
│
├── notebooks/
│   ├── NetraScanAI.ipynb
│   ├── 01_dataset_pipeline.ipynb.ipynb
│   └── 02_vit_training.ipynb
│
├── data/
│   └── metadata/
│
├── results/
│   └── vit_v4/
│       ├── experiment_config.json
│       ├── test_metrics.csv
│       ├── test_predictions.csv
│       ├── training_history.csv
│       ├── vit_v4_accuracy_curve.png
│       └── vit_v4_confusion_matrix.png
│
└── docs/
    └── NetraScan_AI_Dataset_Engineering_Summary.docx
```

Large datasets, model checkpoints, local environments, and generated application reports are excluded from Git tracking.

## 10. Installation

### Requirements

- Python 3.11
- Windows or another compatible Python environment
- NVIDIA GPU recommended for faster inference
- Sufficient system RAM and storage for PyTorch and model dependencies

Clone the repository:

```bash
git clone https://github.com/ShobhitJain2197/NetraScan.AI.git
cd NetraScan.AI
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

For GPU acceleration, install a PyTorch build compatible with your NVIDIA driver and CUDA environment.

### Model Checkpoints

Trained model checkpoints are not included in the standard Git repository.

To run inference, the required checkpoints must be placed in the model directory expected by `src/inference.py`.

The source code alone is not sufficient to reproduce the trained inference results without these weights.

## 11. Running the Application

After configuring the Python environment and supplying the trained checkpoints, start the application:

```bash
python app.py
```

Open the local Gradio URL displayed in the terminal, normally:

```text
http://127.0.0.1:7860
```

### Portable SSD Deployment

A separately prepared portable edition includes:

- Python 3.11 runtime
- Installed dependencies
- Trained model checkpoints
- Application source code
- Windows batch launcher

The portable application can be started using:

```text
START_NETRASCAN.bat
```

The portable edition has been functionally tested on a desktop RTX 3080 and a separate laptop GPU system.

**Note:** The GitHub source repository does not include the complete portable Python environment or trained model weights. Its launcher requires the corresponding portable runtime layout.

## 12. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Vision architectures | timm |
| User interface | Gradio |
| Explainability | Grad-CAM |
| Image processing | OpenCV, Pillow |
| Numerical operations | NumPy |
| Data analysis | pandas |
| Visualization | Plotly, Matplotlib |
| Hardware acceleration | NVIDIA CUDA |

## 13. Limitations

NetraScan.AI is an academic research prototype with the following limitations:

- Evaluation primarily uses a curated internal dataset.
- Performance on external clinical datasets has not been established.
- Predictions may be unreliable for images captured under unfamiliar lighting, camera, or acquisition conditions.
- Confidence scores are not equivalent to clinically calibrated probabilities.
- Grad-CAM maps do not constitute clinical evidence.
- The system does not replace ophthalmological examinations.

## 14. Future Scope

Planned or potential improvements include:

- Evaluation on additional independent datasets.
- Broader testing with real-world eye photographs.
- Improved confidence calibration and uncertainty handling.
- Expanded explainability comparisons.
- Integration of an LLM-assisted reporting module.
- Further optimization for lower-resource hardware.
- Additional research documentation and model comparisons.

## 15. Author & Contributors

### Project Creator

**Mr. SHOBHIT JAIN**

**Role:** Project Creator, AI/ML Developer & Researcher

**GitHub:** [ShobhitJain2197](https://github.com/ShobhitJain2197)

## 16. Research Disclaimer

**NetraScan.AI is developed strictly for academic research, education, and experimental evaluation.**

It is not a certified medical device and must not be used as a substitute for professional diagnosis, medical advice, or clinical decision-making.

---

**NetraScan.AI — Multi-Transformer Cataract Classification & Explainable AI**

**Created by Mr. SHOBHIT JAIN**

*© 2026 NetraScan.AI. All rights reserved by the respective rights holders.*