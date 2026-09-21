# Design Decisions

## CascadeLP Architecture

```
Input pair (id1, id2)
  │
  ├─ Tier 0: id1 == id2  ──→  predict 1  (trivial self-loop)
  │
  ├─ Tier 1: StructuralClassifier  ──→  max(proba) ≥ tier1_threshold?  ──→  output
  │            cold-start pairs (not in graph) → skip straight to Tier 2
  │
  ├─ Tier 2: PosClassifier  ──→  max(proba) ≥ tier2_threshold?  ──→  output
  │
  └─ Tier 3: EmbeddingClassifier  ──→  output (no threshold)
```

Each tier outputs a confidence score; if `max(proba) ≥ threshold`, the pair is classified and exits. Otherwise it
cascades to the next tier.

## Key Decisions

**LogisticRegression over SVC for core tiers**
SVC RBF is O(n²–n³) — infeasible on the full training set. LogReg trains in seconds. A real `SvmClassifier` (RBF SVC
on a stratified subsample) exists as a standalone baseline.

**POS features at Tier 2**
Reproduces the DSAA 2023 baseline result with POS+RF and is fast at inference — no transformer needed.

**Stream nodes.tsv in chunks**
`load_nodes_for_ids()` keeps only the IDs needed by the current pair set in memory.

**Filter self-loops from training**
Nearly all train self-loops are positive. They would inflate metrics if left in the main tiers. Handled separately in
Tier 0.

**Three independent tiers**
Each tier is trained on the full training set; no cascading labels between tiers.


**`compute_heuristics(G, pairs)` argument order**
Graph is the first argument. Reversed args silently KeyError because networkx treats the dataframe as an adjacency
lookup.

**Copy figures to `paper/figures/`**
Figures in `outputs/figures/` are ephemeral. Copying to `paper/figures/` (tracked in git) keeps the thesis
self-contained at every commit.

## Framing

CascadeLP is framed primarily as a **cost-aware diagnostic**: measuring which tier resolves which pairs,
cross-referenced against a data-driven difficulty label, rather than claiming headline accuracy improvements. See
`plan.md` for full scope and status.
