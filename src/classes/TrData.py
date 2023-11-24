from typing import NamedTuple

import pandas as pd

class TrData(NamedTuple):
    customer_id: pd.DataFrame
    churn_label: pd.DataFrame
    transformed_df: pd.DataFrame