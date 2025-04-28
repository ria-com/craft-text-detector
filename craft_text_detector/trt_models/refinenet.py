from .base import TrtRunner
from typing import Tuple
import numpy as np


class RefineNetTrtInference:
    """
    Вхід = два тензори:
        craft_y   ― (N, H/2, W/2, 2)  float32|float16
        features  ― (N, 32,  H/2, W/2)
    Повертає refined_affinity.
    """

    def __init__(self, engine_path: str):
        self.runner = TrtRunner(engine_path)

    def __call__(
        self, craft_y: np.ndarray, craft_features: np.ndarray
    ) -> np.ndarray:
        outputs = self.runner(
            {
                "craft_output_y": craft_y.astype(np.float16)
                if craft_y.dtype == np.float32
                else craft_y,
                "craft_output_features": craft_features.astype(np.float16)
                if craft_features.dtype == np.float32
                else craft_features,
            }
        )
        return outputs["refined_affinity"]
