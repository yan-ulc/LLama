import math 
from typing import Optional, Tuple
from dataclasses import dataclass

import torch 
from torch import nn
import numpy as np
np.random.seed()

@dataclass
class ModelArgs:
    dim: int = 64
    n_layers: int = 8
    n_heads: int = 8
    