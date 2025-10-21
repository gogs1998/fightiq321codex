"""
Kelly Criterion bet sizing helpers.
"""

import numpy as np


def kelly_fraction(p: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Compute Kelly fraction for decimal odds.

    p: win probability
    odds: decimal odds
    Returns fraction of bankroll to bet; negatives -> 0.
    """
    b = odds - 1.0
    f = (p * (b + 1) - 1) / b
    return np.clip(np.maximum(f, 0.0), 0.0, 1.0)

