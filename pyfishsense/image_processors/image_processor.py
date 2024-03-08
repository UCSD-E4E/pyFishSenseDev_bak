from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class ImageProcessor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def load_and_process(self, path: Path):
        raise NotImplementedError

    @abstractmethod
    def process(self, raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError
