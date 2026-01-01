import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import RobustScaler

from core.utils import PlotHistory
from core.logging import LoggerFactory
from core.model import AutoEncoderModelArchitecture


logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)


BASE_PATH = Path(__file__).resolve().parent
FEATURES_PATH = BASE_PATH / "dataset" / "features.csv"
ARTIFACT_PATH = BASE_PATH / "artifacts"

SCALER_PATH = ARTIFACT_PATH / "feature-scaler.pkl"
MODEL_PATH = ARTIFACT_PATH / "autoencoder-model.keras"
PLOT_PATH = ARTIFACT_PATH / "loss-plot.png"

os.makedirs(ARTIFACT_PATH, exist_ok=True)
logger.debug("Artifacts directory ensured")


def main() -> None:
    logger.info("Starting autoencoder training pipeline")

    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
    logger.info("Artifact directory ready: %s", ARTIFACT_PATH)

    logger.info("Loading features from %s", FEATURES_PATH)
    df = pd.read_csv(FEATURES_PATH)

    if "shop_id" not in df.columns:
        logger.warning("'shop_id' column not found in features")
    else:
        df.drop(columns="shop_id", inplace=True)
        logger.info("'shop_id' column dropped")

    logger.info("Feature matrix shape: %s", df.shape)

    logger.info("Applying RobustScaler")
    scaler = RobustScaler()
    x = scaler.fit_transform(df)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Feature scaler saved to %s", SCALER_PATH)

    logger.info(
        "Initializing autoencoder | input_space=%d | latent_space=%d | seed=%d",
        x.shape[1], 10, 42
    )

    model = AutoEncoderModelArchitecture(
        seed=42,
        input_space=x.shape[1],
        latent_space=5
    )

    model.summary()

    logger.info("Starting model training")

    history = model.fit(
        x=x,
        y=x,
        epochs=30,
        batch_size=300,
        validation_split=0.35,
        shuffle=True
    )

    logger.info("Training completed")

    logger.info("Saving training history plot")
    plot = PlotHistory(history=history).plot_history()
    plot.write_image(PLOT_PATH)

    logger.info("Saving trained autoencoder model to %s", MODEL_PATH)
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
