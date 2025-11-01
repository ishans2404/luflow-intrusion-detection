from __future__ import annotations

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import logger

hiddenimports = [
    "xgboost",
    "xgboost.core",
    "xgboost.callback",
    "xgboost.training",
    "xgboost.sklearn",
    "xgboost.dask",
    "xgboost.compat",
]

datas = []

spec = importlib.util.find_spec("xgboost")
if spec and spec.submodule_search_locations:
    package_dir = Path(spec.submodule_search_locations[0])
    datas.append((str(package_dir), "xgboost"))
else:
    logger.warning("Failed to locate xgboost package directory; runtime import may fail")
