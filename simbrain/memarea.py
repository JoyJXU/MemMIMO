import torch
from typing import Iterable, Optional, Union
import json


class Area(torch.nn.Module):
    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        memristor_info_dict: dict = {},
        length_row: float = 0.0,
        length_col: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.shape = shape
        self.length_row = length_row
        self.length_col = length_col

        self.array_area = self.cal_area()


    def cal_area(self):
        array_area_value = self.length_row * self.length_col
        return array_area_value