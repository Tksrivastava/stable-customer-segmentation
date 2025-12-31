# setting up logger
from pathlib import Path
from core.logging import LoggerFactory

logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)

# loading .env
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=f"{Path(__file__).resolve().parents[2]}\.env")
logger.info(".env loaded")

# connecting to kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# creating folder for dataset
os.makedirs(os.getenv("KAGGLE_DOWNLOAD_PATH"), exist_ok=True)
logger.info("Directory created for downloading raw data")


class FetchFromKaggle(object):
    def __init__(self):
        self.logger = logger
        self.api = KaggleApi()
        self.api.authenticate()
        self.logger.info("Connection established to Kaggle")

    def download(self):
        # downloading dataset
        self.api.dataset_download_files(
            dataset=os.getenv("KAGGLE_DATASET"),
            path=os.getenv("KAGGLE_DOWNLOAD_PATH"),
            unzip=True,
            force=True,
        )
        self.logger.info("Raw data downloaded")
