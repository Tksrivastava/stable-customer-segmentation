# Stable Customer Segmentation in FMCG Retail  
### A Representation-Learning–First Clustering System

This repository contains a **production-oriented case study** on building **stable, meaningful customer clusters** in an FMCG retail setting.

Instead of clustering directly on high-dimensional engineered features, this project demonstrates how **learning compact latent representations using an Autoencoder** can dramatically improve cluster stability, health, and downstream usability.

The focus is **not** on algorithm novelty, but on **system behavior under real-world constraints**:
- Feature growth
- Temporal drift
- Eligibility filtering
- Reproducibility
- Inference consistency

---

## Motivation

In real FMCG systems, traditional clustering pipelines frequently fail due to:

- Degrading cluster quality as feature space grows  
- Extremely small or fragmented clusters  
- False homogeneity at aggregate levels  
- Continuous cluster reshuffling (“cluster explosion”) during retraining  

This repository explores a practical alternative:

> **Learn a stable behavioral representation first, then cluster.**

---

## High-Level Approach

1. Filter retailers using tenure and activity-based eligibility rules  
2. Engineer behavior-driven retail features (seasonality, stability, entropy, growth)  
3. Establish a raw-feature clustering baseline  
4. Train a deterministic Autoencoder for representation learning  
5. Cluster retailers in latent space using density-based methods  
6. Compare raw vs latent clustering on:
   - Cluster count
   - Noise ratio
   - Size distribution
   - Stability characteristics  

---

## Dataset

The project uses a **synthetic-style, anonymized FMCG retail dataset** published on Kaggle.

- Monthly aggregated sales quantities  
- `(shop_id, product_id, year, month)` granularity  
- Privacy-preserving transformations applied  
- No real-world identifiers or customer-level data  

**Dataset link:**  
https://www.kaggle.com/datasets/tanulkumarsrivastava/sales-dataset

This dataset is intended for **research, learning, and benchmarking**, not operational deployment.

---

## Repository Structure

```

.
├── core/
│   ├── features.py              # Feature engineering logic
│   ├── model.py                 # Autoencoder architecture
│   ├── utils.py                 # Data fetching, filtering, plotting utilities
│   └── logging.py               # Centralized logging configuration
│
├── artifacts/
│   ├── autoencoder-model.keras  # Trained autoencoder
│   ├── feature-scaler.pkl       # Feature scaler
│   ├── hdbscan-latent.pkl       # Latent-space clustering model
│   ├── hdbscan-raw.pkl          # Raw-feature clustering model
│   ├── cluster_insights.csv     # Cluster-level statistics
│   └── loss-plot.png            # Autoencoder training curve
│
├── feature-preparation-pipeline.py
├── autoencoder-training-pipeline.py
├── clustering-training-pipeline.py
├── clustering-inference-pipeline.py
└── README.md

````

---

## Pipelines Overview

### 1. Feature Preparation
```bash
python feature-preparation-pipeline.py
````

* Downloads dataset from Kaggle
* Applies eligibility filtering
* Generates retailer-level clustering features

---

### 2. Autoencoder Training

```bash
python autoencoder-training-pipeline.py
```

* Scales features using RobustScaler
* Trains a deterministic autoencoder
* Saves model, scaler, and training diagnostics

---

### 3. Clustering Training

```bash
python clustering-training-pipeline.py
```

* Clusters retailers in:

  * Raw feature space (baseline)
  * Latent representation space
* Saves clustering artifacts

---

### 4. Clustering Inference & Insights

```bash
python clustering-inference-pipeline.py
```

* Assigns cluster labels and strengths
* Generates cluster-level statistics for analysis

---

## Key Results (Summary)

| Method                  | Clusters | Noise Ratio |
| ----------------------- | -------- | ----------- |
| Raw Feature Clustering  | 2        | ~99.6%      |
| Latent Space Clustering | 4        | ~21.5%      |

Latent-space clustering produces:

* More balanced clusters
* Significantly lower noise
* Better behavioral separation
* Improved stability across retraining

---

## Design Principles

* **Systems over notebooks**
* **Stability over one-time accuracy**
* **Reproducibility over speed**
* **Behavioral features over raw aggregates**
* **Deferred interpretability, not ignored interpretability**

---

## Disclaimer

This project is a **technical case study**.
Cluster labels and interpretations are illustrative and do not represent real retailers or business entities.

---

## Related Writing

A detailed technical write-up explaining the motivation, system design, and results of this project is available as a Medium article:

👉 *[Link to Medium article]*

---

Here is the **clean, correct, copy-paste ready Markdown** version you should add to your `README.md`.

This version is **professional, clear, and production-appropriate**.

## How to Use

### 1. Create a virtual environment
```bash
python -m venv .venv
````

### 2. Run the full pipeline

```bash
run.bat
```
