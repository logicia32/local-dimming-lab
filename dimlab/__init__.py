"""local-dimming-lab: 液晶 local dimming の効果と副作用を自作モデルで描く小道具。"""

from .core import DimResult, metrics, reference, simulate
from .scenes import SCENES

__all__ = ["DimResult", "metrics", "reference", "simulate", "SCENES"]
