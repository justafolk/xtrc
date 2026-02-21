from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    level = os.getenv("AINAV_LOG_LEVEL", "INFO").upper()
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
