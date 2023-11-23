from typing import NamedTuple, Union


class Outlier(NamedTuple):
    idx: int
    outlier_val: Union[float, int]