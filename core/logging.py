import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


class LoggerFactory:
    """
    Centralized logger factory for the project.
    Ensures consistent formatting, handlers, and log locations.
    """

    def __init__(
        self,
        level: int = logging.INFO,
        log_to_file: bool = True,
        log_dir: str = "logs",
        log_file_name: str = "app.log",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 3,
    ):
        self.level = level
        self.log_to_file = log_to_file
        self.log_dir = log_dir
        self.log_file_name = log_file_name
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        self.formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self.project_root = Path(__file__).resolve().parents[1]

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a configured logger.

        Args:
            name (str): Logger name (usually __name__)

        Returns:
            logging.Logger
        """

        logger = logging.getLogger(name)
        logger.setLevel(self.level)

        # Prevent duplicate handlers
        if logger.handlers:
            return logger

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        logger.addHandler(console_handler)

        # File handler
        if self.log_to_file:
            log_path = self.project_root / self.log_dir
            log_path.mkdir(exist_ok=True)

            file_handler = RotatingFileHandler(
                filename=log_path / self.log_file_name,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
            )
            file_handler.setFormatter(self.formatter)
            logger.addHandler(file_handler)

        return logger
