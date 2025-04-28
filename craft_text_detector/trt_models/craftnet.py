from .base import TrtRunner
from typing import Tuple
import numpy as np


class CraftTrtInference:
    """
    Зручна обгортка: бере BGR/RGB float32|float16 з нормалізацією в
    [0,1], (N,3,H,W) → повертає ('regions','affinity').
    """

    def __init__(self, engine_path: str):
        self.runner = TrtRunner(engine_path)

    def __call__(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if images.ndim != 4:
            raise ValueError("Очікую тензор (N,3,H,W)")
        outputs = self.runner({"images": images})
        return outputs["regions"], outputs["affinity"]
