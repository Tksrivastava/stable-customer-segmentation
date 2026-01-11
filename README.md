# **Stable Customer Segmentation in FMCG Retail**

### *A Representation-Learning-First Clustering System*

> **Author:** Tanul Kumar Srivastava
> **Medium Deep-Dive:**
> [https://medium.com/aimonks/why-most-retail-customer-clusters-collapse-in-production-and-how-i-fixed-mine-122a412ceccf](https://medium.com/aimonks/why-most-retail-customer-clusters-collapse-in-production-and-how-i-fixed-mine-122a412ceccf)

---

## 📌 What this project solves

In real FMCG and retail systems, customer segmentation **breaks down in production** due to:

* Feature explosion over time
* Data drift across retraining cycles
* Unstable cluster assignments
* Extremely high noise ratios
* Poor reproducibility

Most clustering systems are built as **one-off experiments**, not **long-running production systems**.

This repository demonstrates a **production-oriented alternative**:

> **Learn stable behavioral representations first → then cluster**

Instead of clustering on raw engineered features, we:

1. Learn compact latent embeddings using a deterministic Autoencoder
2. Cluster in latent space using HDBSCAN
3. Compare raw-feature vs latent-space behavior across stability, noise, and size distribution

This is **not** about inventing new algorithms.
It is about building **clustering that survives real-world retraining cycles**.

---

## 🧠 System Architecture

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
Cluster Assignments + Strengths
        │
        ▼
Cluster Analytics & Stability Metrics
```

---

## 📊 Dataset

The project uses a **synthetic-style, anonymized FMCG retail dataset** published on Kaggle.

**Dataset:**
[https://www.kaggle.com/datasets/tanulkumarsrivastava/sales-dataset](https://www.kaggle.com/datasets/tanulkumarsrivastava/sales-dataset)

**Structure**

* Monthly aggregated sales
* `(shop_id, product_id, year, month)`
* Privacy-preserving transformations
* No real customer or retailer data

This dataset is intended for:

* Algorithm benchmarking
* Feature engineering
* Clustering system design

---

## 🗂 Repository Structure

```
stable-customer-segmentation/
│
├── core/
│   ├── features.py        # Retailer behavior feature engineering
│   ├── model.py           # Autoencoder architecture
│   ├── utils.py           # Data loading, filtering, plots
│   └── logging.py         # Centralized logging
│
├── artifacts/             # Trained models and outputs
│   ├── autoencoder-model.keras
│   ├── feature-scaler.pkl
│   ├── hdbscan-latent.pkl
│   ├── hdbscan-raw.pkl
│   ├── cluster_insights.csv
│   └── loss-plot.png
│
├── feature-preparation-pipeline.py
├── autoencoder-training-pipeline.py
├── clustering-training-pipeline.py
├── clustering-inference-pipeline.py
│
├── run.bat
└── README.md
```

This is structured like a **real ML system**, not a notebook dump:

* Reusable modules
* Explicit pipelines
* Model artifacts
* Inference scripts

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Tksrivastava/stable-customer-segmentation.git
cd stable-customer-segmentation
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv .venv
```

Activate:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / Mac**

```bash
source .venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run full pipeline

```bash
run.bat
```

This executes:

1. Feature generation
2. Autoencoder training
3. Raw + latent clustering
4. Cluster analytics

All models and outputs are saved into `/artifacts`.

---

## 🔁 Pipelines Explained

### **1. Feature Preparation**

```bash
python feature-preparation-pipeline.py
```

What happens:

* Downloads dataset from Kaggle
* Filters retailers using:

  * Minimum activity
  * Tenure thresholds
* Builds **behavioral features**:

  * Stability
  * Entropy
  * Growth
  * Seasonality
  * Volume consistency

These are not naive aggregates — they are **representation-learning-ready signals**.

---

### **2. Autoencoder Training**

```bash
python autoencoder-training-pipeline.py
```

What happens:

* Robust scaling
* Deterministic Autoencoder training
* Loss tracking
* Model persistence

Outputs:

* Trained encoder/decoder
* Scaler
* Training curve

This creates a **stable latent behavioral space**.

---

### **3. Clustering Training**

```bash
python clustering-training-pipeline.py
```

Two clustering strategies are trained:

| Strategy             | Purpose             |
| -------------------- | ------------------- |
| Raw Feature HDBSCAN  | Baseline            |
| Latent Space HDBSCAN | Production approach |

Both are saved for comparison and inference.

---

### **4. Inference & Analytics**

```bash
python clustering-inference-pipeline.py
```

Generates:

* Cluster labels
* Membership strength
* Cluster sizes
* Noise ratios
* Distribution reports

---

## 📈 Key Results

| Method       | Clusters | Noise  |
| ------------ | -------- | ------ |
| Raw Features | 2        | ~99.6% |
| Latent Space | 4        | ~21.5% |

This is **not** a tuning trick — it is a **systemic effect** of learning representations before clustering.

---

## 🧩 Why Autoencoder + HDBSCAN?

Retail data is:

* High dimensional
* Sparse
* Noisy
* Non-linear

Autoencoders:

* Compress signal
* Remove noise
* Learn invariant behavioral structure

HDBSCAN:

* Handles density variation
* Naturally models noise
* Works well in learned manifolds

This combination is **production-grade for segmentation**.

---

## 🧪 Making this Production Ready

This repository is already **architecture-ready** for real systems.

Here is how you would evolve it:

### 🔹 Add MLflow

You can log:

* Autoencoder versions
* Clustering models
* Feature scalers
* Metrics (noise %, cluster count, stability)

Example:

```python
mlflow.log_model(autoencoder, "encoder")
mlflow.log_metric("noise_ratio", noise)
```

---

### 🔹 Add Feature Store / DB

Replace:

```python
df = load_from_csv()
```

With:

* Snowflake
* BigQuery
* Postgres
* Feast Feature Store

This allows:

* Backfills
* Time-travel
* Reproducible training

---

### 🔹 Add Batch Inference

The inference pipeline already supports:

* New retailers
* New months
* Re-scoring into existing clusters

You only need:

* A scheduler (Airflow / Dagster)
* A data sink (warehouse / API)

---

## 🧠 What this repo demonstrates

This project shows how to think like a **production data scientist**:

* Not “best silhouette score”
* Not “best k-means”
* But **long-term stability, drift resistance, and cluster usability**

This is exactly how **real retail segmentation platforms** are built.

---

## 📚 Deep Technical Explanation

Read the full system breakdown here:
👉 [https://medium.com/aimonks/why-most-retail-customer-clusters-collapse-in-production-and-how-i-fixed-mine-122a412ceccf](https://medium.com/aimonks/why-most-retail-customer-clusters-collapse-in-production-and-how-i-fixed-mine-122a412ceccf)

---