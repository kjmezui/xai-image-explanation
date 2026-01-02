# src/utils/logger.py

import logging
import os
from typing import Optional


def get_logger(
    name: str = "xai_project",
    log_dir: str = "logs",
    log_file: str = "training.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Crée un logger qui écrit à la fois dans la console et dans un fichier.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Éviter les handlers dupliqués si on rappelle get_logger
    if logger.handlers:
        return logger

    # Format commun
    fmt = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Handler fichier
    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
