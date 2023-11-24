from typing import NamedTuple

import numpy as np

class NpData(NamedTuple):
    ohe_np: np.ndarray
    churn_np: np.ndarray