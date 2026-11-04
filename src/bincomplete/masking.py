import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class Mask:
    core: np.ndarray
    hidden: np.ndarray
    absent: bool = False


def source_rng(seed, source_id):
    digest = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))
