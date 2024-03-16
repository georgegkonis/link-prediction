import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


@dataclass
class EvalResult:
    macro_f1: float
    auc_roc: float
    cold_start_macro_f1: float | None = None
    cold_start_count: int = 0
    latency_ms: float | None = None
    extra: dict = field(default_factory=dict)

    def __str__(self):
        lines = [
            f'  Macro F1          : {self.macro_f1:.4f}',
            f'  AUC-ROC           : {self.auc_roc:.4f}',
        ]
        if self.cold_start_macro_f1 is not None:
            lines.append(f'  Cold-start F1     : {self.cold_start_macro_f1:.4f}  (n={self.cold_start_count})')
        if self.latency_ms is not None:
            lines.append(f'  Latency           : {self.latency_ms:.1f} ms')
        return '\n'.join(lines)


def cold_start_mask(pairs: pd.DataFrame, G: nx.Graph) -> np.ndarray:
    """Boolean mask — True where id1 and id2 share zero common neighbours."""
    mask = []
    for _, row in pairs.iterrows():
        u, v = row['id1'], row['id2']
        if u == v:
            mask.append(False)
            continue
        if not G.has_node(u) or not G.has_node(v):
            mask.append(True)
            continue
        mask.append(len(list(nx.common_neighbors(G, u, v))) == 0)
    return np.array(mask)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
    cs_mask: np.ndarray | None = None,
    latency_ms: float | None = None,
) -> EvalResult:
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = roc_auc_score(y_true, y_scores) if y_scores is not None else float('nan')

    cs_f1, cs_count = None, 0
    if cs_mask is not None and cs_mask.sum() > 0:
        cs_count = int(cs_mask.sum())
        cs_f1 = f1_score(y_true[cs_mask], y_pred[cs_mask], average='macro', zero_division=0)

    return EvalResult(
        macro_f1=macro_f1,
        auc_roc=auc,
        cold_start_macro_f1=cs_f1,
        cold_start_count=cs_count,
        latency_ms=latency_ms,
    )


@contextmanager
def timer():
    """Context manager that yields a list; after the block list[0] holds elapsed ms."""
    result = []
    t0 = time.perf_counter()
    yield result
    result.append((time.perf_counter() - t0) * 1000)
