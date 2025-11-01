from __future__ import annotations
from typing import Any, Iterable, Union
import numpy as np
import json
import pathlib

ArrayLike = Union[np.ndarray, Iterable[float], float, int]

class EnergyModel:
    """Callable that returns an energy in eV."""
    def __call__(self, desc: Any) -> float:
        raise NotImplementedError

class LinearModel(EnergyModel):
    """Toy linear model: energy = w · x + b (stub for demos)."""
    def __init__(self, w: ArrayLike, b: float = 0.0):
        self.w = np.asarray(w, float).ravel()
        self.b = float(b)

    def __call__(self, desc: Any) -> float:
        x = np.asarray(desc, float).ravel()
        if x.shape[0] != self.w.shape[0]:
            raise ValueError(f"Descriptor length {x.shape[0]} != weight length {self.w.shape[0]}")
        return float(self.w.dot(x) + self.b)

def load_linear_model(path: Union[str, pathlib.Path]) -> LinearModel:
    """Load weights from JSON file with keys 'w' and optional 'b'."""
    p = pathlib.Path(path)
    data = json.loads(p.read_text())
    return LinearModel(data["w"], data.get("b", 0.0))
