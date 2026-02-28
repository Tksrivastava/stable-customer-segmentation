import pandas as pd
from pathlib import Path

from core.logging import LoggerFactory
from core.features import RetailClusteringFeatureBuilder
from core.utils import FetchFromKaggle, CustomerEligibilityFilter


logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)


FILE_PATH = Path(__file__).resolve().parents[0]
DATASET_PATH = FILE_PATH / "dataset" / "shop-sales-data.csv"
FEATURES_OUTPUT_PATH = FILE_PATH / "dataset" / "features.csv"


def main() -> None:
    """
    Execute the feature generation pipeline.
    """
    logger.info("Starting retail clustering feature generation pipeline")

    logger.info("Ensuring dataset availability via Kaggle fetch")
    FetchFromKaggle().download()

    logger.info("Loading raw transactional data from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)

    logger.info(
        "Raw data loaded | rows=%d | cols=%d",
        df.shape[0],
        df.shape[1]
    )

    logger.info("Applying customer eligibility criteria")

    eligibility_filter = CustomerEligibilityFilter(
        year_col="year",
        month_col="month",
        customer_id_col="shop_id",
        min_tenure_years=1,
        min_avg_active_months_per_year=6
    )

    eligible_customers = eligibility_filter.filter_customers(df=df)

    logger.info(
        "Eligibility filtering completed | eligible_customers=%d",
        len(eligible_customers)
    )

    df = df.loc[df["shop_id"].isin(eligible_customers)].copy()

    logger.info(
        "Filtered dataset prepared | rows=%d",
        df.shape[0]
    )

    logger.info("Building clustering features")

    builder = RetailClusteringFeatureBuilder(df=df)
    features = builder.build_features()

    logger.info(
        "Feature generation completed | rows=%d | cols=%d",
        features.shape[0],
        features.shape[1]
    )

    logger.info("Writing features to %s", FEATURES_OUTPUT_PATH)

    features.to_csv(FEATURES_OUTPUT_PATH, index=False)

    logger.info("Feature pipeline completed successfully")


if __name__ == "__main__":
    main()
