import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import RobustScaler
from hdbscan import HDBSCAN

from core.logging import LoggerFactory
from core.model import AutoEncoderModelArchitecture


logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)


BASE_PATH = Path(__file__).resolve().parents[0]

DATASET_PATH = BASE_PATH / "dataset" / "features.csv"

ARTIFACT_PATH = BASE_PATH / "artifacts"
ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)

AE_MODEL_PATH = ARTIFACT_PATH / "autoencoder-model.keras"
FEATURE_SCALER_PATH = ARTIFACT_PATH / "feature-scaler.pkl"

LATENT_CLUSTER_MODEL_PATH = ARTIFACT_PATH / "hdbscan-latent.pkl"
RAW_CLUSTER_MODEL_PATH = ARTIFACT_PATH / "hdbscan-raw.pkl"
RAW_SCALER_PATH = ARTIFACT_PATH / "raw-feature-scaler.pkl"


def main() -> None:
    logger.info("Starting clustering pipeline")

    logger.info("Loading trained autoencoder from %s", AE_MODEL_PATH)
    model = AutoEncoderModelArchitecture.load(AE_MODEL_PATH)

    logger.info("Loading feature scaler from %s", FEATURE_SCALER_PATH)
    with open(FEATURE_SCALER_PATH, "rb") as f:
        feature_scaler = pickle.load(f)

    logger.info("Loading dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)

    df_features = df.drop(columns="shop_id")
    logger.info("Feature matrix shape: %s", df_features.shape)

    logger.info("Scaling input features")
    X_scaled = feature_scaler.transform(df_features)

    logger.info("Generating latent representations")
    Z = model.get_encoded_input(X_scaled)

    logger.info("Latent space shape: %s", Z.shape)

    logger.info("Clustering latent space using HDBSCAN")

    latent_clusterer = HDBSCAN(
        min_cluster_size=50,
        prediction_data=True
    )

    latent_labels = latent_clusterer.fit_predict(Z)

    logger.info(
        "Latent clustering completed | clusters=%d | noise_ratio=%.3f",
        len(set(latent_labels)) - (1 if -1 in latent_labels else 0),
        (latent_labels == -1).mean()
    )

    # Save latent clustering model
    with open(LATENT_CLUSTER_MODEL_PATH, "wb") as f:
        pickle.dump(latent_clusterer, f)

    logger.info("Saved latent HDBSCAN model to %s", LATENT_CLUSTER_MODEL_PATH)

    logger.info("Running baseline clustering on raw features")

    raw_scaler = RobustScaler()
    X_raw_scaled = raw_scaler.fit_transform(df_features)

    raw_clusterer = HDBSCAN(
        min_cluster_size=50,
        prediction_data=True
    )

    raw_labels = raw_clusterer.fit_predict(X_raw_scaled)

    logger.info(
        "Raw feature clustering completed | clusters=%d | noise_ratio=%.3f",
        len(set(raw_labels)) - (1 if -1 in raw_labels else 0),
        (raw_labels == -1).mean()
    )

    # Save raw artifacts
    with open(RAW_CLUSTER_MODEL_PATH, "wb") as f:
        pickle.dump(raw_clusterer, f)

    with open(RAW_SCALER_PATH, "wb") as f:
        pickle.dump(raw_scaler, f)

    logger.info("Saved raw HDBSCAN model to %s", RAW_CLUSTER_MODEL_PATH)
    logger.info("Saved raw feature scaler to %s", RAW_SCALER_PATH)

if __name__ == "__main__":
    main()