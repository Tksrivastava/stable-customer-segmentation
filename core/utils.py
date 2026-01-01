import os
from typing import List
import tensorflow as tf
from pathlib import Path
import plotly.graph_objects as go

import pandas as pd
from dotenv import load_dotenv

from core.logging import LoggerFactory


logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)


FILE_PATH = Path(__file__).resolve().parents[1]
ENV_PATH = f"{FILE_PATH}/.env"
load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env")

class FetchFromKaggle:
    """
    Handles authentication and dataset download from Kaggle.

    Responsibilities:
    - Authenticate with Kaggle using environment credentials
    - Download and unzip datasets
    - Ensure consistent local storage path

    Assumes:
    - KAGGLE_USERNAME and KAGGLE_KEY are present in .env
    - KAGGLE_DATASET is defined in .env
    """

    def __init__(self):
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.logger = logger
        self.api = KaggleApi()

        self.logger.info("Authenticating with Kaggle API")
        self.api.authenticate()

        self.save_path = FILE_PATH / "dataset"
        self._create_download_path()

        self.logger.info(
            "Kaggle connection established. Download path: %s", self.save_path
        )

    def _create_download_path(self) -> None:
        """
        Creates the dataset download directory if it does not exist.
        """
        os.makedirs(self.save_path, exist_ok=True)
        self.logger.debug("Dataset directory ensured at %s", self.save_path)

    def download(self) -> None:
        """
        Downloads and extracts the Kaggle dataset specified in the environment.
        """
        dataset_name = os.getenv("KAGGLE_DATASET")
        if not dataset_name:
            raise ValueError("KAGGLE_DATASET not found in environment variables")

        self.logger.info("Starting dataset download: %s", dataset_name)

        self.api.dataset_download_files(
            dataset=dataset_name,
            path=self.save_path,
            unzip=True,
            force=True,
        )

        self.logger.info("Dataset downloaded and extracted successfully")


class CustomerEligibilityFilter:
    """
    Filters customers based on tenure and activity stability.

    This class is intended to identify 'good' or 'stable' customers
    before downstream tasks such as clustering or segmentation.

    Filtering criteria:
    - Minimum tenure (in years)
    - Minimum average number of active months per year
    """

    def __init__(
        self,
        year_col: str,
        month_col: str,
        customer_id_col: str,
        min_tenure_years: float,
        min_avg_active_months_per_year: float,
    ):
        self.year_col = year_col
        self.month_col = month_col
        self.customer_id_col = customer_id_col
        self.min_tenure_years = min_tenure_years
        self.min_avg_active_months_per_year = min_avg_active_months_per_year

        logger.info(
            "CustomerEligibilityFilter initialized | min_tenure_years=%s, min_avg_active_months_per_year=%s",
            min_tenure_years,
            min_avg_active_months_per_year,
        )

    def filter_customers(self, df: pd.DataFrame) -> List[str]:
        """
        Filters eligible customers from the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input transactional or aggregated dataset containing
            customer, year, and month information.

        Returns
        -------
        List[str]
            List of customer IDs satisfying eligibility criteria.
        """

        logger.debug("Starting customer eligibility filtering")
        logger.debug("Input dataframe shape: %s", df.shape)

        # Deduplicate at (customer, year, month) level
        df_unique = df[
            [self.customer_id_col, self.year_col, self.month_col]
        ].drop_duplicates()

        logger.debug("After deduplication shape: %s", df_unique.shape)

        # Monthly activity per customer-year
        yearly_activity = (
            df_unique
            .groupby([self.customer_id_col, self.year_col], sort=False)
            .size()
            .rename("active_months")
            .reset_index()
        )

        # Aggregate to customer-level metrics
        customer_metrics = (
            yearly_activity
            .groupby(self.customer_id_col, sort=False)
            .agg(
                total_active_months=("active_months", "sum"),
                avg_active_months_per_year=("active_months", "mean"),
            )
            .reset_index()
        )

        logger.debug(
            "Computed customer metrics for %d customers",
            customer_metrics.shape[0],
        )

        # Apply eligibility criteria
        eligible_customers = customer_metrics.loc[
            (customer_metrics["total_active_months"] >= 12 * self.min_tenure_years)
            & (
                customer_metrics["avg_active_months_per_year"]
                >= self.min_avg_active_months_per_year
            ),
            self.customer_id_col,
        ]

        logger.info(
            "Eligible customers identified: %d out of %d",
            eligible_customers.shape[0],
            customer_metrics.shape[0],
        )

        return eligible_customers.tolist()
    
class PlotHistory(object):
    """
Utility class for visualizing training history of TensorFlow / Keras models.

This class provides a lightweight wrapper around a `tf.keras.callbacks.History`
object and generates interactive loss curves using Plotly. It is intended for
quick inspection of training dynamics such as convergence behavior and
overfitting.

Currently supported metrics:
- Training loss (`loss`)
- Validation loss (`val_loss`), if available

The visualization is rendered as an interactive Plotly figure with epoch-wise
traces, suitable for exploratory analysis in notebooks or local development
environments.

Parameters
----------
history : tf.keras.callbacks.History, optional
    History object returned by `model.fit`. Must contain a `history` attribute
    with recorded loss values.

Notes
-----
- This utility assumes that the model was compiled with a loss function.
- Validation loss is plotted only if `val_loss` is present in the history.
- The class does not perform input validation and will raise an error if an
    invalid or incomplete History object is provided.
- Intended for visualization only; it does not return numerical results.
"""
    def __init__(self, history: tf.keras.callbacks.History = None):
        self.history = history
    def plot_history(self):
        hist = self.history.history
        epochs = list(range(1, len(hist['loss']) + 1))

        fig = go.Figure()

        # Training loss
        fig.add_trace(go.Scatter(x=epochs, y=hist['loss'], mode='lines+markers', name='Train Loss'))

        # Validation loss (if exists)
        if 'val_loss' in hist:
            fig.add_trace(go.Scatter(x=epochs, y=hist['val_loss'], mode='lines+markers', name='Validation Loss'))

        fig.update_layout(title='History', xaxis_title='Epoch',
                        yaxis_title='Loss', template='plotly_white', hovermode='x unified')
        fig.show()
        return fig