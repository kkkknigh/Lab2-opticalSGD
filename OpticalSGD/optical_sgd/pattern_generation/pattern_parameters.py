"""Pattern parameter container."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PatternParameters:
    values: np.ndarray

    def copy(self) -> "PatternParameters":
        return PatternParameters(values=np.array(self.values, copy=True))
