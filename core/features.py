import time
import numpy as np
import pandas as pd

from core.logging import LoggerFactory

logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)

class RetailClusteringFeatureBuilder:
    """
    RetailClusteringFeatureBuilder

    Builds stable, behavior-driven retailer features for clustering
    in FMCG retail systems.

    Logging:
    --------
    This class does not configure logging internally.
    A pre-configured logger must be injected.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the feature builder.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transactional data with columns:
            [shop_id, year, month, product_id, sales_qty]
        """
        self.df = df.copy()
        self.monthly = None
        self.logger = logger

        self.logger.info(
            "RetailClusteringFeatureBuilder initialized | rows=%d | cols=%d",
            self.df.shape[0],
            self.df.shape[1]
        )
        
    def _build_monthly_table(self) -> None:
        if self.monthly is not None:
            self.logger.debug("Monthly table already built, skipping.")
            return

        self.logger.info("Building monthly aggregation table...")
        start = time.time()

        self.monthly = (
            self.df
            .groupby(
                ["shop_id", "year", "month"],
                observed=True,
                sort=False
            )
            .agg(
                monthly_sales_qty=("sales_qty", "sum"),
                monthly_unique_products=("product_id", "nunique")
            )
            .reset_index()
        )

        self.logger.info(
            "Monthly aggregation completed | rows=%d | time=%.2fs",
            len(self.monthly),
            time.time() - start
        )

    def _build_monthly_averages(self) -> pd.DataFrame:
        self.logger.info("Computing monthly seasonality features...")
        start = time.time()

        avg_sales = (
            self.monthly
            .groupby(["shop_id", "month"], sort=False)["monthly_sales_qty"]
            .mean()
            .round(2)
            .unstack(fill_value=0)
            .add_prefix("avg_sales_qty_month_")
            .reset_index()
        )

        avg_spread = (
            self.monthly
            .groupby(["shop_id", "month"], sort=False)["monthly_unique_products"]
            .mean()
            .round(2)
            .unstack(fill_value=0)
            .add_prefix("avg_unique_products_month_")
            .reset_index()
        )

        self.logger.info(
            "Seasonality features built | shops=%d | time=%.2fs",
            avg_sales.shape[0],
            time.time() - start
        )

        return avg_sales.merge(avg_spread, on="shop_id", how="inner")

    def _build_cv_features(self) -> pd.DataFrame:
        self.logger.info("Computing CV (stability) features...")
        start = time.time()

        cv_df = (
            self.monthly
            .groupby("shop_id")
            .agg(
                mean_monthly_sales_qty=("monthly_sales_qty", "mean"),
                std_monthly_sales_qty=("monthly_sales_qty", "std"),
                mean_monthly_unique_products=("monthly_unique_products", "mean"),
                std_monthly_unique_products=("monthly_unique_products", "std")
            )
            .assign(
                cv_monthly_sales_qty=lambda x: np.where(
                    x["mean_monthly_sales_qty"] > 0,
                    x["std_monthly_sales_qty"] / x["mean_monthly_sales_qty"],
                    0.0
                ),
                cv_monthly_unique_products=lambda x: np.where(
                    x["mean_monthly_unique_products"] > 0,
                    x["std_monthly_unique_products"] / x["mean_monthly_unique_products"],
                    0.0
                )
            )
            .reset_index()[[
                "shop_id",
                "cv_monthly_sales_qty",
                "cv_monthly_unique_products"
            ]]
        )

        self.logger.info(
            "CV features built | shops=%d | time=%.2fs",
            cv_df.shape[0],
            time.time() - start
        )

        return cv_df

    def _build_entropy_feature(self) -> pd.DataFrame:
        self.logger.info("Computing monthly sales entropy...")
        start = time.time()

        monthly = self.monthly.copy()

        monthly["total_sales_qty"] = (
            monthly.groupby("shop_id")["monthly_sales_qty"]
            .transform("sum")
        )

        monthly["sales_probability"] = np.where(
            monthly["total_sales_qty"] > 0,
            monthly["monthly_sales_qty"] / monthly["total_sales_qty"],
            0.0
        )

        monthly["entropy_component"] = np.where(
            monthly["sales_probability"] > 0,
            -monthly["sales_probability"] * np.log(monthly["sales_probability"]),
            0.0
        )

        entropy_df = (
            monthly
            .groupby("shop_id")
            .agg(
                raw_entropy=("entropy_component", "sum"),
                active_months=("month", "nunique")
            )
            .assign(
                monthly_sales_entropy=lambda x: np.where(
                    x["active_months"] > 1,
                    x["raw_entropy"] / np.log(x["active_months"]),
                    0.0
                )
            )
            .reset_index()[["shop_id", "monthly_sales_entropy"]]
        )

        self.logger.info(
            "Entropy feature built | shops=%d | time=%.2fs",
            entropy_df.shape[0],
            time.time() - start
        )

        return entropy_df

    def _build_yoy_growth_features(self) -> pd.DataFrame:
        self.logger.info("Computing YoY median growth features...")
        start = time.time()

        yearly = (
            self.monthly
            .groupby(["shop_id", "year"], sort=False)
            .agg(
                yearly_sales_qty=("monthly_sales_qty", "sum"),
                yearly_avg_unique_products=("monthly_unique_products", "mean")
            )
            .reset_index()
            .sort_values(["shop_id", "year"])
        )

        yearly["yoy_sales_growth"] = (
            yearly.groupby("shop_id")["yearly_sales_qty"]
            .pct_change()
        )

        yearly["yoy_spread_growth"] = (
            yearly.groupby("shop_id")["yearly_avg_unique_products"]
            .pct_change()
        )

        yoy_df = (
            yearly
            .groupby("shop_id")
            .agg(
                median_yoy_sales_growth=("yoy_sales_growth", "median"),
                median_yoy_spread_growth=("yoy_spread_growth", "median")
            )
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .reset_index()
        )

        self.logger.info(
            "YoY growth features built | shops=%d | time=%.2fs",
            yoy_df.shape[0],
            time.time() - start
        )

        return yoy_df

    def build_features(self) -> pd.DataFrame:
        """
        Build the full retailer feature table.

        Returns
        -------
        pd.DataFrame
            One row per shop_id with all clustering features.
        """
        self.logger.info("Starting full feature build pipeline...")
        start = time.time()

        self._build_monthly_table()

        features = (
            self._build_monthly_averages()
            .merge(self._build_cv_features(), on="shop_id", how="left")
            .merge(self._build_entropy_feature(), on="shop_id", how="left")
            .merge(self._build_yoy_growth_features(), on="shop_id", how="left")
        )

        self.logger.info(
            "Feature build completed | shops=%d | cols=%d | total_time=%.2fs",
            features.shape[0],
            features.shape[1],
            time.time() - start
        )

        return features
