"""
Structured logging utility with rotating file handler.
Uses Python's built-in logging with a consistent format for the entire pipeline.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


_loggers: dict = {}  # module-level cache


def get_logger(name: str, log_file: Optional[str] = None,
               level: str = "INFO", max_bytes: int = 10_485_760,
               backup_count: int = 3) -> logging.Logger:
    """
    Return a cached, configured logger.

    Parameters
    ----------
    name        : Logger name (typically __name__).
    log_file    : Path to rotating log file. None → console only.
    level       : Logging level string.
    max_bytes   : Rotating file max size.
    backup_count: Number of backup files to keep.
    """
    global _loggers
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _loggers[name] = logger
    return logger


def configure_from_config(cfg: dict) -> None:
    """Bootstrap root logger from config.yaml logging section."""
    log_cfg = cfg.get("logging", {})
    root = get_logger(
        "vpd",
        log_file=log_cfg.get("log_file"),
        level=log_cfg.get("level", "INFO"),
        max_bytes=log_cfg.get("max_bytes", 10_485_760),
        backup_count=log_cfg.get("backup_count", 3),
    )
    root.info("D-SAAT logging initialised.")
