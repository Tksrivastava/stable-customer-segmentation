# Stable Customer Segmentation in FMCG Retail

### *A Representation-Learning-First Clustering System*

> **Author:** Tanul Kumar Srivastava
> **Medium Deep-Dive:** [Why Most Retail Customer Clusters Collapse in Production — and How I Fixed Mine](https://medium.com/aimonks/why-most-retail-customer-clusters-collapse-in-production-and-how-i-fixed-mine-122a412ceccf)

---

## What This Project Solves

In real FMCG and retail systems, customer segmentation **breaks down in production** due to:

- Feature explosion over time
- Data drift across retraining cycles
- Unstable cluster assignments
- Extremely high noise ratios
- Poor reproducibility

Most clustering systems are built as **one-off experiments**, not **long-running production systems**.

This repository demonstrates a **production-oriented alternative**:

> **Learn stable behavioral representations first → then cluster**

Instead of clustering on raw engineered features, the system:

1. Learns compact latent embeddings via a deterministic Autoencoder
2. Clusters in latent space using HDBSCAN
3. Compares raw-feature vs. latent-space behavior across stability, noise, and cluster size distribution

This is **not** about inventing new algorithms — it is about building **clustering that survives real-world retraining cycles**.

---

## System Architecture

```
Raw FMCG Transactions
        │
        ▼
Eligibility Filtering
        │
        ▼
Retailer Feature Engineering
        │
        ├───────────────► Raw Feature Clustering (Baseline)
        │
        ▼
Robust Scaling
        │
        ▼
Autoencoder (Representation Learning)
        │
        ▼
Latent Space Embeddings
        │
        ▼
HDBSCAN Clustering
        │
        ▼
Cluster Assignments + Membership Strengths
        │
        ▼
Cluster Analytics & Stability Metrics
```

---

## Dataset

This project uses a **synthetic-style, anonymized FMCG retail dataset** published on Kaggle.

**Dataset:** [tanulkumarsrivastava/sales-dataset](https://www.kaggle.com/datasets/tanulkumarsrivastava/sales-dataset)

The dataset contains monthly aggregated sales with the schema `(shop_id, product_id, year, month)`, with privacy-preserving transformations applied — no real customer or retailer data is included.

Intended use cases:
- Algorithm benchmarking
- Feature engineering experiments
- Clustering system design

---

## Repository Structure

```
stable-customer-segmentation/
├── core/
│   ├── __init__.py
│   ├── features.py               # RetailClusteringFeatureBuilder
│   ├── logging.py                # LoggerFactory
│   ├── model.py                  # AutoEncoderModelArchitecture
│   └── utils.py
├── autoencoder-training-pipeline.py
├── clustering-inference-pipeline.py
├── clustering-training-pipeline.py
├── feature-preparation-pipeline.py
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirement.txt
├── run.sh
└── README.md
```

> **Note:** The `/artifacts` folder (trained models, scalers, outputs) is not included in this repository due to file size. Running the full pipeline via `run.sh` regenerates all artifacts locally.

This is structured like a **real ML system**, not a notebook dump — reusable modules, explicit pipelines, and dedicated inference scripts.

---

## Getting Started

### Prerequisites

- Python `>=3.11, <3.12`
- A [Kaggle API key](https://www.kaggle.com/docs/api) for dataset download

### 1. Clone the repository

```bash
git clone https://github.com/Tksrivastava/stable-customer-segmentation.git
cd stable-customer-segmentation
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your Kaggle credentials:

```env
KAGGLE_USERNAME="your_username"
KAGGLE_KEY="your_api_key"
KAGGLE_DATASET="tanulkumarsrivastava/sales-dataset/"
```

### 3. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirement.txt
pip install -e .
```

### 5. Run the full pipeline

```bash
bash run.sh
```

This executes all four stages in order and saves models and outputs into `/artifacts`.

---

## Pipeline Stages

### Stage 1 — Feature Preparation

```bash
python feature-preparation-pipeline.py
```

- Downloads the dataset from Kaggle
- Filters retailers by minimum activity and tenure thresholds
- Builds behavioral features via `RetailClusteringFeatureBuilder`:

| Feature | Description |
|---|---|
| Stability | Consistency of sales volume over time |
| Entropy | Diversity/randomness in purchasing patterns |
| Growth | Trend direction and rate |
| Seasonality | Periodic demand patterns |
| Volume Consistency | Month-over-month variance in quantities |

These are representation-learning-ready signals, not naive aggregates.

---

### Stage 2 — Autoencoder Training

```bash
python autoencoder-training-pipeline.py
```

Trains a deterministic `AutoEncoderModelArchitecture`:

- Symmetric feed-forward architecture (encoder → latent → decoder)
- Linear latent space to preserve geometric structure for distance-based methods
- Encoder-side L2 regularization to prevent identity mapping
- Batch normalization in the encoder for training stability
- Layer normalization on the latent space to control scale drift
- MSE reconstruction objective with early stopping

Outputs saved to `/artifacts`:
- `autoencoder-model.keras` — full trained autoencoder
- `feature-scaler.pkl` — robust scaler fitted on training data
- `loss-plot.png` — training/validation loss curve

> GPU is disabled by design for deterministic, reproducible CPU execution.

---

### Stage 3 — Clustering Training

```bash
python clustering-training-pipeline.py
```

Trains two HDBSCAN models for direct comparison:

| Strategy | Input | Purpose |
|---|---|---|
| Raw Feature HDBSCAN | Engineered features | Baseline |
| Latent Space HDBSCAN | Autoencoder embeddings | Production approach |

Both models are saved to `/artifacts` for use in inference.

---

### Stage 4 — Inference & Analytics

```bash
python clustering-inference-pipeline.py
```

Generates full cluster analytics:
- Cluster labels and membership strengths
- Cluster size distributions
- Noise ratios per strategy
- Comparative distribution reports

Outputs saved to `/artifacts`:
- `cluster_predictions.csv`
- `cluster_insights.csv`

---

## Key Results

| Method | Clusters | Noise Ratio |
|---|---|---|
| Raw Features | 2 | ~99.6% |
| Latent Space | 4 | ~21.5% |

This is **not** a hyperparameter tuning trick — it is a **systemic effect** of decoupling representation learning from clustering. The latent space forces the model to learn smooth, structured behavioral manifolds that HDBSCAN can meaningfully partition.

---

## Why Autoencoder + HDBSCAN?

FMCG retail data is high-dimensional, sparse, noisy, and non-linear. This combination addresses that directly:

**Autoencoder** — compresses signal, removes noise, and learns invariant behavioral structure that survives retraining cycles.

**HDBSCAN** — handles variable cluster density, models noise as a first-class concept, and works well in learned latent manifolds where distance is more meaningful than in raw feature space.

---

## Extending to Production

The architecture is already designed with production evolution in mind.

**Experiment Tracking — MLflow**

```python
mlflow.log_model(autoencoder, "encoder")
mlflow.log_metric("noise_ratio", noise_ratio)
mlflow.log_metric("cluster_count", n_clusters)
```

**Feature Store / Database**

Replace CSV loading with a proper data layer:
- Snowflake / BigQuery / Postgres for warehouse-backed features
- Feast for time-travel and backfill-safe feature retrieval

**Scheduled Batch Inference**

The inference pipeline already supports new retailers and new months. Add:
- A scheduler (Airflow / Dagster / Prefect)
- A data sink (warehouse table / REST API)

---

## Core Dependencies

| Package | Version | Role |
|---|---|---|
| TensorFlow | 2.15.1 | Autoencoder training |
| hdbscan | 0.8.38.post2 | Density-based clustering |
| scikit-learn | 1.4.2 | Preprocessing, metrics |
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.26.4 | Numerical operations |
| plotly + kaleido | 5.24.1 / 0.2.1 | Visualizations |
| kaggle | 1.5.16 | Dataset download |
| python-dotenv | 1.0.1 | Environment config |

Full dependency list in `requirement.txt`.

---

## License

MIT License — see [LICENSE](./LICENSE) for details.

---

## Deep Technical Write-Up

For a full breakdown of the system design, failure modes of naive clustering, and the production reasoning behind this architecture:

👉 [Why Most Retail Customer Clusters Collapse in Production — and How I Fixed Mine](https://medium.com/aimonks/why-most-retail-customer-clusters-collapse-in-production-and-how-i-fixed-mine-122a412ceccf)
