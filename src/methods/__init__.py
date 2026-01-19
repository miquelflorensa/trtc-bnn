# Proposed methods for real-to-categorical transformations
#
# Methods:
#   1. LL-Softmax: Locally Linearized Softmax (GMA-based)
#   2. MM-Softmax: Moment-Matching Softmax (log-normal matching)
#   3. MM-Remax: Moment-Matching Remax (ReLU-based, numerically stable)
#
# Monte Carlo baselines:
#   - MC-Softmax: Monte Carlo sampling for Softmax
#   - MC-Remax: Monte Carlo sampling for Remax

# Analytical methods
from .ll_softmax import LLSoftmax, ll_softmax, deterministic_softmax
from .mm_softmax import MMSoftmax, mm_softmax
from .mm_remax import MMRemax, mm_remax, deterministic_remax

# Monte Carlo baselines
from .mc_softmax import MCSoftmax, mc_softmax, mc_softmax_expected
from .mc_remax import MCRemax, mc_remax, mc_remax_expected

__all__ = [
    # LL-Softmax (Method 1)
    "LLSoftmax",
    "ll_softmax",
    # MM-Softmax (Method 2)
    "MMSoftmax",
    "mm_softmax",
    "deterministic_softmax",
    # MM-Remax (Method 3)
    "MMRemax",
    "mm_remax",
    "deterministic_remax",
    # MC baselines
    "MCSoftmax",
    "mc_softmax",
    "mc_softmax_expected",
    "MCRemax",
    "mc_remax",
    "mc_remax_expected",
]
