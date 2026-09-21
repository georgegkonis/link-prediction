"""
Fetch real article text for the crawled wikipedia node set from the
wikimedia/wikipedia Hugging Face dataset (simple English config), matching by
exact title. Streams the dataset so the full ~300MB+ config need not be
downloaded to disk first.

Reads:  <output>/titles.json        (from build_from_wikidump.py)
Writes: <output>/nodes.tsv          (id, text — same schema as data/raw/dsaa/nodes.tsv)
        <output>/text_fetch_stats.json (match-rate provenance, for thesis macros)

Usage:
    python -m scripts.data.fetch_wikipedia_text --output data/raw/wikipedia
"""
import argparse
import json
import pathlib
import time

import pandas as pd
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/raw/wikipedia')
    parser.add_argument('--config', default='20231101.simple')
    args = parser.parse_args()
    out_dir = pathlib.Path(args.output)

    id_to_title = json.loads((out_dir / 'titles.json').read_text())
    title_to_id = {t.replace('_', ' '): i for i, t in id_to_title.items()}
    wanted = set(title_to_id.keys())
    print(f'Looking for {len(wanted):,} titles in wikimedia/wikipedia[{args.config}]...')

    ds = load_dataset('wikimedia/wikipedia', args.config, split='train', streaming=True)
    found = {}
    t0 = time.time()
    for i, row in enumerate(ds):
        if row['title'] in wanted:
            found[row['title']] = row['text']
            if len(found) >= len(wanted):
                break
        if i % 200_000 == 0:
            print(f'  scanned {i:,} rows, found {len(found):,}/{len(wanted):,}, '
                  f'{time.time()-t0:.0f}s elapsed', flush=True)
    print(f'Scan done: found {len(found):,}/{len(wanted):,} titles in {time.time()-t0:.0f}s')

    rows = []
    for title, node_id in title_to_id.items():
        rows.append((int(node_id), found.get(title, '')))
    nodes_df = pd.DataFrame(rows, columns=['id', 'text']).set_index('id').sort_index()
    nodes_df.to_csv(out_dir / 'nodes.tsv', sep='\t')

    n_missing = int((nodes_df['text'].str.len() == 0).sum())
    print(f'Wrote {out_dir}/nodes.tsv — {len(nodes_df)} nodes, {n_missing} with empty text')

    stats = {
        'config': args.config,
        'n_titles_wanted': len(wanted),
        'n_titles_matched': len(found),
        'match_pct': 100 * len(found) / len(wanted),
        'n_nodes_empty_text': n_missing,
    }
    (out_dir / 'text_fetch_stats.json').write_text(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
