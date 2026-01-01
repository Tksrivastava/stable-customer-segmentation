import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hdbscan import approximate_predict

from core.logging import LoggerFactory
from core.model import AutoEncoderModelArchitecture


logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)


BASE_PATH = Path(__file__).resolve().parents[0]

DATASET_PATH = BASE_PATH / "dataset" / "features.csv"

ARTIFACT_PATH = BASE_PATH / "artifacts"

AE_MODEL_PATH = ARTIFACT_PATH / "autoencoder-model.keras"
FEATURE_SCALER_PATH = ARTIFACT_PATH / "feature-scaler.pkl"

LATENT_CLUSTER_MODEL_PATH = ARTIFACT_PATH / "hdbscan-latent.pkl"
RAW_CLUSTER_MODEL_PATH = ARTIFACT_PATH / "hdbscan-raw.pkl"
RAW_SCALER_PATH = ARTIFACT_PATH / "raw-feature-scaler.pkl"

OUTPUT_PREDICTIONS_PATH = ARTIFACT_PATH / "cluster_predictions.csv"
OUTPUT_CLUSTER_STATS_PATH = ARTIFACT_PATH / "cluster_insights.csv"


def main() -> None:
    logger.info("Starting clustering inference pipeline")

    logger.info("Loading autoencoder model")
    model = AutoEncoderModelArchitecture.load(AE_MODEL_PATH)

    logger.info("Loading feature scaler")
    with open(FEATURE_SCALER_PATH, "rb") as f:
        feature_scaler = pickle.load(f)

    logger.info("Loading HDBSCAN models")
    with open(LATENT_CLUSTER_MODEL_PATH, "rb") as f:
        latent_clusterer = pickle.load(f)

    with open(RAW_CLUSTER_MODEL_PATH, "rb") as f:
        raw_clusterer = pickle.load(f)

    logger.info("Loading raw feature scaler")
    with open(RAW_SCALER_PATH, "rb") as f:
        raw_scaler = pickle.load(f)

    logger.info("Loading dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)

    if "shop_id" not in df.columns:
        raise ValueError("'shop_id' column missing from dataset")

    df_features = df.drop(columns="shop_id")
    logger.info("Feature matrix shape: %s", df_features.shape)

    logger.info("Predicting latent-space clusters")

    X_scaled = feature_scaler.transform(df_features)
    Z = model.get_encoded_input(X_scaled)

    latent_labels, latent_strengths = approximate_predict(
        latent_clusterer,
        Z
    )

    df["latent_cluster"] = latent_labels
    df["latent_cluster_strength"] = latent_strengths

    logger.info(
        "Latent prediction completed | clusters=%d | noise_ratio=%.3f",
        len(set(latent_labels)) - (1 if -1 in latent_labels else 0),
        (latent_labels == -1).mean()
    )

    logger.info("Predicting raw-feature clusters")

    X_raw_scaled = raw_scaler.transform(df_features)

    raw_labels, raw_strengths = approximate_predict(
        raw_clusterer,
        X_raw_scaled
    )

    df["raw_cluster"] = raw_labels
    df["raw_cluster_strength"] = raw_strengths

    logger.info(
        "Raw prediction completed | clusters=%d | noise_ratio=%.3f",
        len(set(raw_labels)) - (1 if -1 in raw_labels else 0),
        (raw_labels == -1).mean()
    )

    df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
    logger.info("Saved cluster predictions to %s", OUTPUT_PREDICTIONS_PATH)

    logger.info("Generating cluster-level insights")

    feature_cols = df_features.columns.tolist()

    insights = []

    for cluster_type in ["latent_cluster", "raw_cluster"]:

        grouped = (
            df
            .groupby(cluster_type)[feature_cols]
            .agg(["mean", "median", "std"])
        )

        grouped.columns = [
            f"{col[0]}_{col[1]}" for col in grouped.columns
        ]

        grouped = grouped.reset_index()

        grouped["cluster_label"] = grouped[cluster_type].apply(
            lambda x: "NOISE (-1)" if x == -1 else f"CLUSTER {x}"
        )

        grouped["cluster_type"] = cluster_type

        grouped["cluster_size"] = (
            df
            .groupby(cluster_type)
            .size()
            .values
        )

        insights.append(grouped)

    cluster_insights_df = pd.concat(insights, ignore_index=True)


    cluster_insights_df.to_csv(OUTPUT_CLUSTER_STATS_PATH, index=False)
    logger.info("Saved cluster insights to %s", OUTPUT_CLUSTER_STATS_PATH)

    logger.info("Clustering inference pipeline completed successfully")

if __name__ == "__main__":
    main()