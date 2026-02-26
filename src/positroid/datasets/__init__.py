"""Dataset registry combining all dataset families."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from positroid.datasets.digits import DIGIT_DATASETS
from positroid.datasets.toy2d import DATASETS as TOY_DATASETS

DATASETS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    **TOY_DATASETS,
    **DIGIT_DATASETS,
}
