import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


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
    """True when either endpoint has no observed non-self neighbour in ``G``.

    Self-pairs are handled separately. This operational cold-start definition
    is intentionally narrower than zero common neighbours: two observed nodes
    may have disjoint neighbourhoods while still carrying structural evidence.
    """
    observed = {u for u in G if any(v != u for v in G[u])}
    return ((pairs['id1'] != pairs['id2']) &
            (~pairs['id1'].isin(observed) | ~pairs['id2'].isin(observed))).to_numpy(dtype=bool)


def zero_common_neighbors_mask(pairs: pd.DataFrame, G: nx.Graph) -> np.ndarray:
    """True for non-self pairs with zero common neighbours.

    Pairs with an endpoint absent from ``G`` are included because they also
    have no observable common neighbour. Use :func:`cold_start_mask` when the
    question is endpoint coverage rather than sparse structural evidence.
    """
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


def structural_groups(pairs: pd.DataFrame, G: nx.Graph) -> np.ndarray:
    """Assign mutually exclusive graph-coverage groups to candidate pairs."""
    observed = {u for u in G if any(v != u for v in G[u])}
    first_unobserved = ~pairs['id1'].isin(observed)
    second_unobserved = ~pairs['id2'].isin(observed)
    groups = np.where(
        zero_common_neighbors_mask(pairs, G),
        'observed_zero_cn',
        'observed_positive_cn',
    ).astype(object)
    groups[(first_unobserved ^ second_unobserved).to_numpy()] = 'one_unobserved'
    groups[(first_unobserved & second_unobserved).to_numpy()] = 'both_unobserved'
    groups[(pairs['id1'] == pairs['id2']).to_numpy()] = 'self_loop'
    return groups


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


def evaluate_by_group(
    y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray,
) -> pd.DataFrame:
    """Per-group n / accuracy / macro-F1, indexed by distinct group value."""
    rows = {}
    for g in pd.unique(group):
        mask = group == g
        rows[g] = {
            'n': int(mask.sum()),
            'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
            'macro_f1': f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0),
        }
    return pd.DataFrame.from_dict(rows, orient='index')


def tier_difficulty_breakdown(
    y_true: np.ndarray, y_pred: np.ndarray, tier_used: np.ndarray, difficulty: pd.Series,
) -> pd.DataFrame:
    """Cross-tab of (tier, difficulty) -> n / accuracy / macro_f1."""
    df = pd.DataFrame({
        'tier': tier_used, 'difficulty': difficulty.values,
        'y_true': y_true, 'y_pred': y_pred,
    })
    rows = {}
    for (tier, diff), group in df.groupby(['tier', 'difficulty']):
        rows[(tier, diff)] = {
            'n': len(group),
            'accuracy': accuracy_score(group['y_true'], group['y_pred']),
            'macro_f1': f1_score(group['y_true'], group['y_pred'], average='macro', zero_division=0),
        }
    return pd.DataFrame.from_dict(rows, orient='index')


@contextmanager
def timer():
    """Context manager that yields a list; after the block list[0] holds elapsed ms."""
    result = []
    t0 = time.perf_counter()
    yield result
    result.append((time.perf_counter() - t0) * 1000)
