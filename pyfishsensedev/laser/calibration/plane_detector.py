from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class PlaneDetector(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
