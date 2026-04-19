import math
import warnings
import torch.nn as nn

# 保持与旧代码 `from .model_utils import *` 兼容：
# 目前 `model_ViLa_MIL.py` 主要依赖 MultiheadAttention / math / warnings。
MultiheadAttention = nn.MultiheadAttention

