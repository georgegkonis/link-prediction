"""Compare DSAA models with the same labelled-pair and graph-edge budget.

Every model is trained on the same stratified sample for a given seed. The
structural graph is rebuilt from positive pairs in that sample only, and each
sampled target edge is masked while its structural features are calculated.
Cached text/POS features are reused because they do not contain edge labels.

The experiment controls training-set size inside DSAA; it does not repair the
dataset's negative-sampling artifact.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import build_graph, load_edges
from src.data.protocol import prepare_split
from src.features.structural import compute_heuristics
from src.models.cascade import CascadeLP
from src.models.svm import PosClassifier, SvmClassifier, TfidfClassifier
from src.utils.metrics import evaluate, timer


MODELS = ('tfidf_lr', 'tfidf_svm', 'pos_rf', 'cascade')


def _load_score(path: Path, column: str, pairs: pd.DataFrame) -> np.ndarray:
    frame = pd.read_csv(path, index_col='id')
    if not frame.index.equals(pairs.index):
        raise ValueError(f'Cached feature IDs do not match DSAA pairs: {path}')
    return frame[column].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=20_000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--raw', default='data/raw/dsaa/train.csv')
    parser.add_argument('--split-directory', default='data/interim/graph_holdout_v1')
    parser.add_argument('--features', default='data/interim/dsaa')
    parser.add_argument(
        '--output',
        default='outputs/predictions/graph_holdout_v1/matched_samples_v2',
    )
    args = parser.parse_args()

    pairs = load_edges(args.raw)
    partition, split_meta = prepare_split(pairs, args.split_directory)
    output = Path(args.output)
    if output.exists():
        protocol_path = output / 'protocol.json'
        metrics_path = output / 'metrics.csv'
        if protocol_path.exists() and metrics_path.exists():
            previous = json.loads(protocol_path.read_text())
            metrics = pd.read_csv(metrics_path)
            expected = {(seed, model) for seed in args.seeds for model in MODELS}
            actual = set(zip(metrics['sample_seed'], metrics['model']))
            if (previous.get('split') == split_meta
                    and previous.get('sample_size') == args.size
                    and previous.get('seeds') == args.seeds
                    and previous.get('models') == list(MODELS)
                    and len(metrics) == len(expected) and actual == expected):
                print(f'Reusing completed matched comparison: {output}')
                return
        raise FileExistsError(f'Existing matched comparison is incomplete or differs: {output}; choose a fresh --output')
    train_idx = np.flatnonzero(partition.to_numpy() == 'train')
    val_idx = np.flatnonzero(partition.to_numpy() == 'val')
    labels = pairs['label'].to_numpy()
    if not 2 <= args.size <= len(train_idx):
        raise ValueError(f'--size must be between 2 and {len(train_idx):,}')

    feature_dir = Path(args.features)
    tfidf = _load_score(feature_dir / 'tfidf_train.csv', 'tfidf_score', pairs)
    sentence = _load_score(feature_dir / 'sentence_emb_train.csv', 'st_score', pairs)
    pos = np.load(feature_dir / 'pos_train.npy', mmap_mode='r')
    if pos.shape != (len(pairs), 72):
        raise ValueError(f'Expected POS cache shape {(len(pairs), 72)}, got {pos.shape}')

    output.mkdir(parents=True)
    protocol = {
        'split': split_meta,
        'dataset': 'DSAA 2023',
        'interpretation': 'matched training size within the artifact-affected dataset',
        'graph_context': 'positive edges from the matched sample only',
        'target_edge_masked': True,
        'sample_size': args.size,
        'seeds': args.seeds,
        'models': list(MODELS),
    }
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')

    rows: list[dict] = []
    for seed in args.seeds:
        sample = train_test_split(
            train_idx,
            train_size=args.size,
            stratify=labels[train_idx],
            random_state=seed,
        )[0]
        pd.Series(pairs.index[sample], name='id').to_csv(
            output / f'sample_{seed}.csv', index=False)

        print(f'[{seed}] computing graph features from {len(sample):,} sampled labels', flush=True)
        with timer() as graph_time:
            graph = build_graph(pairs.iloc[sample])
            sample_structural = compute_heuristics(
                graph, pairs.iloc[sample], exclude_target_edge=True)
            val_structural = compute_heuristics(
                graph, pairs.iloc[val_idx], exclude_target_edge=True)

        for name in MODELS:
            print(f'[{seed}] fitting {name}', flush=True)
            with timer() as fit_time:
                if name == 'cascade':
                    model = CascadeLP(random_state=seed)
                    model.fit(
                        sample_structural,
                        pos[sample],
                        sentence[sample],
                        labels[sample],
                        pairs.iloc[sample],
                    )
                elif name == 'tfidf_lr':
                    model = TfidfClassifier(random_state=seed)
                    model.fit(tfidf[sample], labels[sample])
                elif name == 'tfidf_svm':
                    model = SvmClassifier(subsample_size=args.size, random_state=seed)
                    model.fit(tfidf[sample], labels[sample])
                else:
                    model = PosClassifier(random_state=seed)
                    model.fit(pos[sample], labels[sample])

            with timer() as inference_time:
                if name == 'cascade':
                    prediction, _, probability = model.predict(
                        val_structural,
                        pos[val_idx],
                        sentence[val_idx],
                        pairs.iloc[val_idx],
                    )
                else:
                    features = pos[val_idx] if name == 'pos_rf' else tfidf[val_idx]
                    proba = model.predict_proba(features)
                    prediction, probability = proba.argmax(axis=1), proba[:, 1]

            result = evaluate(labels[val_idx], prediction, probability)
            row = {
                'model': name,
                'sample_seed': seed,
                'n_train': len(sample),
                'n_val': len(val_idx),
                'graph_label_budget': len(sample),
                'graph_positive_edges': graph.number_of_edges(),
                'graph_preparation_ms': graph_time[0],
                'macro_f1': result.macro_f1,
                'auc_roc': result.auc_roc,
                'fit_ms': fit_time[0],
                'inference_ms': inference_time[0],
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
