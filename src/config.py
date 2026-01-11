"""Configuration and logging module for TerraCover.

This module provides centralized configuration loading and a standardized
logging interface used throughout the application.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Module-level constants
_CONFIG_FILE = "config.yaml"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


def _create_logger(name: str = "TerraCover") -> logging.Logger:
    """Create and configure the application logger.

    Args:
        name: Logger identifier.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATE_FORMAT))
        logger.addHandler(handler)

    return logger


_logger = _create_logger()


def log(message: str, level: str = "INFO") -> None:
    """Log a message at the specified level.

    Args:
        message: The message to log.
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.
    """
    level_map = {
        "DEBUG": _logger.debug,
        "INFO": _logger.info,
        "WARNING": _logger.warning,
        "ERROR": _logger.error,
    }
    log_fn = level_map.get(level.upper(), _logger.info)
    log_fn(message)


def _load_config(path: str = _CONFIG_FILE) -> Dict[str, Any]:
    """Load and validate the application configuration.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigurationError: If the file is missing or malformed.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        log(f"Configuration loaded: {path}")
        return config
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Configuration parse error: {e}")


# Load configuration at module import
try:
    CONFIG: Dict[str, Any] = _load_config()
except ConfigurationError as e:
    log(str(e), level="ERROR")
    sys.exit(1)

# Export commonly used configuration sections
LANDCOVER: Dict[str, Dict] = CONFIG["landcover_classes"]
PHYSICS: Dict[str, Dict] = CONFIG["physics"]
