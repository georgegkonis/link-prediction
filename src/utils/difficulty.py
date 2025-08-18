import numpy as np
import pandas as pd

DIFFICULTY_LEVELS = ['trivial_self_loop', 'trivial_high_cn', 'trivial_high_textsim', 'hard']


def pick_thresholds(
    y: np.ndarray, cn: np.ndarray, tfidf: np.ndarray, fpr: float = 0.01,
) -> tuple[float, float]:
    """
    Data-driven thresholds: the value below which `1 - fpr` of NEGATIVE pairs fall.
    Avoids hand-picked magic numbers while keeping the trivial-pair criterion
    tied to how separable the negative class actually is.
    """
    neg = y == 0
    cn_threshold = float(np.nanpercentile(cn[neg], 100 * (1 - fpr)))
    tfidf_threshold = float(np.nanpercentile(tfidf[neg], 100 * (1 - fpr)))
    return cn_threshold, tfidf_threshold


def label_difficulty(
    pairs: pd.DataFrame,
    cn: np.ndarray,
    tfidf: np.ndarray,
    cn_threshold: float,
    tfidf_threshold: float,
) -> pd.Series:
    """
    Per-pair difficulty label aligned to `pairs.index`, one of DIFFICULTY_LEVELS.
    Priority: self_loop > high_cn > high_textsim > hard — a pair can satisfy
    more than one criterion, and self-loops/CN take precedence over text
    similarity since they're cheaper structural signals.
    """
    self_loop = (pairs['id1'] == pairs['id2']).values
    high_cn = np.nan_to_num(cn, nan=0.0) > cn_threshold
    high_textsim = np.nan_to_num(tfidf, nan=0.0) > tfidf_threshold

    labels = np.full(len(pairs), 'hard', dtype=object)
    labels[high_textsim] = 'trivial_high_textsim'
    labels[high_cn] = 'trivial_high_cn'
    labels[self_loop] = 'trivial_self_loop'

    return pd.Series(labels, index=pairs.index, name='difficulty')


def trivial_mask(labels: pd.Series) -> np.ndarray:
    return (labels != 'hard').values
