# Rethinking Foundation Model Adaptation for Dense Regression in Medical Imaging

### [Anonymous MICCAI 2026 Submission #4917]

**TL;DR:** We show that standard foundation model adaptation protocols (freezing layers, shape-preserving losses) degrade performance by 24-27% for dense medical regression. Our gradient-preserving loss achieves 19.04 μm MAE, outperforming structured protocols by 27%.

---

## 🎯 Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Pre-trained Model
```bash
# Our best model (Gradient-Loss, 19.04 μm MAE)
wget [ANONYMIZED_LINK]/gradient_loss_best.pth -P checkpoints/
```

### 3. Run Inference on GRAPE
```bash
python scripts/evaluate.py \
    --config configs/gradient_loss.yaml \
    --checkpoint checkpoints/gradient_loss_best.pth \
    --data_path ./data/GRAPE \
    --output results/grape_predictions.csv
```
**Expected output:** MAE ≈ 19.88 μm, σ(pred) ≈ 22.60 μm

---

## 📊 Main Results (from Paper)

### Table 1: Multi-Backbone Performance and Metric Collapse Analysis
| Backbone | Pre-train | Protocol | MAE ($\mu m$) ↓ | Pearson R ↑ | Fairness Gap ↓ | $\Delta$ vs. Naive ↓ | $\sigma_{pred}$ ($\mu m$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ImageNet (ViT-B/16)** | Classification | Naive | 19.79 | 0.661 | 1.60 | - | 12.4 |
| | | Aggressive | 19.92 | 0.661 | 1.72 | +0.7% | 12.2 |
| | | Structured | 24.68 | 0.649 | 1.85 | +24.7% | 10.8 |
| **DepthAny (V2-Small)** | Depth (Metric) | Naive | 19.93 | 0.661 | 1.64 | - | 12.5 |
| | | Aggressive | 19.88 | 0.662 | 1.70 | +0.3% | 12.3 |
| | | Structured | 24.75 | 0.645 | 1.82 | +24.2% | 10.6 |
| **RETFound (ViT-L/16)** | MAE Semantic | Naive | 19.52 | 0.662 | 1.68 | - | 12.3 |
| | | Aggressive | 19.25 | 0.676 | 1.79 | -1.4% | 12.3 |
| | | **Gradient-Loss (Ours)** | **19.04\***| **0.676** | 2.40 | **-2.5%** | **11.8** |
| | | Structured | 24.80 | 0.458 | **1.57\***| +27.0% | 4.8 |

<br>

### Table 2: Sector-wise RNFL Regression Results on GRAPE Dataset (Cross-Modality)
| Protocol | Global MAE ↓ | Sup. MAE ↓ | Nas. MAE ↓ | Inf. MAE ↓ | Temp. MAE ↑ | $\sigma_{pred}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RETFound (MAE Only)** | 20.12 | 20.45 | 18.90 | 22.34 | 18.79 | 0.82 (Collapsed) |
| **Ours (Grad + MAE)** | **19.88** | **19.61** | **18.56** | **22.12** | **19.25** | **12.15 (Preserved)** |

**Key Finding:** Structured protocols cause "template overfitting" (σ=4.8 μm), while our gradient loss preserves anatomical diversity (σ=11.8 μm).

### Table 2: Ablation Study
| Configuration | Frozen (%) | MAE (μm) | σ(pred) (μm) |
| :--- | :--- | :--- | :--- |
| Aggressive | 0% | 19.25 | 12.3 |
| freeze_4 | 17% | 23.07 | 12.0 |
| freeze_8 | 33% | 23.57 | 11.7 |
| freeze_12 | 50% | 24.80 | 11.4 |

**Linear penalty:** $MAE$ = 19.25 + 3.6 x (%frozen/10), $R^2=0.98$

---

## 🏗️ Architecture
```text
Input SLO Image (224×224) 
    ↓
RETFound Backbone (ViT-L/16)
    ↓
Projection Head (3-layer MLP → 56×56 spatial map)
    ↓
Radial Aggregator (360° circular sampling)
    ↓
RNFL Head (MLP: 360 → 512 → 360)
    ↓
RNFL Thickness Profile (360 points)
```
**Key Innovation:** Radial Aggregator enforces geometric prior for retinal structure.

---

## 🔧 Reproducing Paper Results

**Option 1: One-Click Reproduction (Recommended)**
```bash
bash scripts/reproduce_paper.sh
```
This will:
* Train baseline models (Naive, Aggressive, Structured)
* Train our Gradient-Loss model
* Run ablation studies (freezing, loss complexity)
* Generate all tables and figures
* Save results to `results/`
*(Time: ~10 hours on A100 GPU)*

**Option 2: Individual Experiments**
```bash
# Train Gradient-Loss Model
python scripts/train.py \
    --config configs/gradient_loss.yaml \
    --data_path ./data/FairFedMed \
    --output_dir checkpoints/gradient_loss

# Evaluate on Test Set
python scripts/evaluate.py \
    --config configs/gradient_loss.yaml \
    --checkpoint checkpoints/gradient_loss_best.pth \
    --data_path ./data/FairFedMed \
    --split test

# Run Ablation Studies
python scripts/ablation.py \
    --study freezing \
    --data_path ./data/FairFedMed
```

## 📦 Datasets

Due to medical data licensing and Data Use Agreements (DUAs), both datasets must be manually downloaded before running the evaluation pipeline. We do not own or distribute these datasets. 

* **FairFedMed (Primary):** 11,539 images (70/15/15 train/val/test split). Demographics labels included. Format: 224×224 fundus images + 360-point RNFL profiles.
  * **Access:** Download the raw `.tar` archive and the master split `.csv` from the [FairFedMed GitHub Repository](https://github.com/Harvard-AI-and-Robotics-Lab/FairFedMed).
  * **Attribution:** Introduced by Li et al. (2025) in *"FairFedMed: Benchmarking Group Fairness in Federated Medical Imaging with FairLoRA"* (IEEE TMI).

* **GRAPE (Cross-Modality Validation):** 243 eyes. Modality: CFP vs SLO. Purpose: Test cross-modality robustness.
  * **Access:** Request access and download the raw dataset archive from the [GRAPE Dataset Figshare](https://springernature.figshare.com/collections/GRAPE_A_multimodal_glaucoma_dataset_of_follow-up_visual_field_and_fundus_images_for_glaucoma_management/6406319/1).
  * **Attribution:** Introduced by Huang et al. (2023) in *"GRAPE: A multi-modal dataset of longitudinal follow-up visual field and fundus images for glaucoma management"* (Scientific Data).

**Data Setup & Preprocessing:**
Once you have downloaded the raw datasets and agreed to their respective DUAs, place them in your local directory and run our preprocessing scripts to align them with our dataloader's expected geometric format:

```bash
# 1. Prepare FairFedMed (Verifies CSV and aligns TAR structure)
python scripts/prepare_fairfedmed.py \
    --input_dir /path/to/your/downloaded/fairfedmed \
    --output_dir ./data/FairFedMed

# 2. Prepare GRAPE (Extracts images and aligns SNIT sector targets)
python scripts/prepare_grape.py \
    --input_dir /path/to/your/downloaded/grape \
    --output_dir ./data/GRAPE
```
*Purpose:* Circularly samples spatial features at 30% radius from optic disc center.

### 2. Gradient-Preserving Loss
```python
from losses import GradientLoss
loss_fn = GradientLoss(lambda_gradient=0.2)
loss = loss_fn(predictions, targets)
```
*Formula:* L = \|T_{pred} - T_{gt}\|_1 + \lambda_g \|\nabla T_{pred} - \nabla T_{gt}\|_1
*Effect:* Preserves anatomical transitions while avoiding template overfitting.

### 3. Variance Metrics
```python
import numpy as np
# Compute prediction variance
sigma_pred = np.std(predictions)  # Should be ~11-12 μm (healthy)

# Template overfitting check
if sigma_pred < 6:
    print("⚠️ Template overfitting detected!")
```

---

## 📈 Expected Results

**On FairFedMed (Test Set)**
* **Gradient-Loss Model:** MAE: 19.04 ± 0.06 μm | Pearson R: 0.676 | σ(pred): 11.8 μm | Fairness Gap: 2.40 μm

**On GRAPE (Cross-Modality)**
* **Gradient-Loss Model:** MAE: 19.88 μm | σ(pred): 22.60 μm
* **Structured Protocol:** MAE: 34.5 μm | σ(pred): 4.8 μm (collapsed)

---

## 🐛 Common Issues
* **CUDA Out of Memory:** Reduce `batch_size` (e.g., to 8 or 4) and increase `gradient_accumulation_steps` in config.
* **Dataset Path Not Found:** Ensure paths in config files use relative paths like `./data/FairFedMed/train` rather than absolute local machine paths.
* **Checkpoint Not Loading:** Run `python scripts/convert_checkpoint.py --input old_checkpoint.pth --output new_checkpoint.pth`

---

## 📊 Generating Paper Figures

```bash
# Figure 2: Template Overfitting Visualization
python notebooks/01_visualize_predictions.ipynb

# Tables 1-3: Main Results
python scripts/generate_tables.py --results_dir results/ --output paper_tables.csv
```

---

## 🔬 Ablation Studies
```bash
# Freezing Strategy
python scripts/ablation.py --study freezing

# Loss Complexity
python scripts/ablation.py --study loss_complexity

# Learning Rates
python scripts/ablation.py --study learning_rate
```

---

## 💾 Pre-trained Models
| Model | Config | MAE (μm) | Download |
| :--- | :--- | :--- | :--- |
| Gradient-Loss (Best) | `configs/gradient_loss.yaml` | 19.04 | `[ANONYMIZED_LINK]` |
| Aggressive Baseline | `configs/baseline.yaml` | 19.25 | `[ANONYMIZED_LINK]` |
| Structured Protocol | `configs/structured.yaml` | 24.80 | `[ANONYMIZED_LINK]` |


## Critical notice
This work is **only for research purpose**, it is **not for clinical use**.


## 📧 Contact
For questions or issues regarding this submission, please use the communication portal on the **MICCAI 2026 CMT Platform** to maintain author anonymity.
