"""
Build a real, connected Wikipedia subgraph from official MediaWiki SQL dumps
(page/linktarget/pagelinks tables) instead of the live API — the live API is
rate-limited too hard on this environment's IP to crawl at any useful scale.

Every edge here is a genuine hyperlink extracted from a real dump; there is no
negative-sampling artifact to correct because we construct labels ourselves in
build_wiki_fresh_dataset.py from these real edges.

Requires (download once, e.g. from https://dumps.wikimedia.org/simplewiki/latest/):
    <dump-dir>/simplewiki-latest-page.sql.gz
    <dump-dir>/simplewiki-latest-linktarget.sql.gz
    <dump-dir>/simplewiki-latest-pagelinks.sql.gz

Writes:
    <output>/titles.json         — page_id -> title mapping (crawled subgraph only)
    <output>/positive_edges.csv  — id1, id2 (real hyperlinks, undirected, deduped,
                                   ids are page_id from the dump)

Usage:
    python -m scripts.data.build_from_wikidump --dump-dir /tmp --seed "Computer_science" --target 8000
"""
import argparse
import json
import pathlib
from collections import deque, defaultdict

from scripts.data.mysql_dump import iter_insert_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-dir', default='/tmp')
    parser.add_argument('--wiki', default='simplewiki')
    parser.add_argument('--seed', default='Computer_science', help='underscored title, as in MediaWiki')
    parser.add_argument('--target', type=int, default=8000)
    parser.add_argument('--output', default='data/wiki_fresh')
    args = parser.parse_args()
    d = pathlib.Path(args.dump_dir)

    print('Parsing page table...')
    id_to_title = {}       # page_id -> title, namespace 0, non-redirect only
    title_to_id = {}
    for row in iter_insert_rows(str(d / f'{args.wiki}-latest-page.sql.gz'), 'page'):
        page_id, ns, title, is_redirect = row[0], row[1], row[2], row[3]
        if ns == '0' and is_redirect == '0':
            id_to_title[page_id] = title
            title_to_id[title] = page_id
    print(f'  {len(id_to_title):,} namespace-0 non-redirect pages')

    print('Parsing linktarget table...')
    lt_to_title = {}        # lt_id -> title, namespace 0 only
    for row in iter_insert_rows(str(d / f'{args.wiki}-latest-linktarget.sql.gz'), 'linktarget'):
        lt_id, ns, title = row[0], row[1], row[2]
        if ns == '0':
            lt_to_title[lt_id] = title
    print(f'  {len(lt_to_title):,} namespace-0 link targets')

    print('Parsing pagelinks table (this is the big one)...')
    adjacency = defaultdict(set)  # page_id -> set(page_id) — real, resolved, namespace-0-to-namespace-0 edges
    n_rows = 0
    for row in iter_insert_rows(str(d / f'{args.wiki}-latest-pagelinks.sql.gz'), 'pagelinks'):
        n_rows += 1
        pl_from, pl_from_ns, pl_target_id = row[0], row[1], row[2]
        if pl_from_ns != '0' or pl_from not in id_to_title:
            continue
        target_title = lt_to_title.get(pl_target_id)
        if target_title is None:
            continue
        target_id = title_to_id.get(target_title)
        if target_id is None or target_id == pl_from:
            continue
        adjacency[pl_from].add(target_id)
        adjacency[target_id]  # ensure key exists for BFS even with in-degree only
    print(f'  {n_rows:,} raw pagelinks rows -> {sum(len(v) for v in adjacency.values()):,} resolved directed edges '
          f'among {len(adjacency):,} pages with at least one link')

    seed_id = title_to_id.get(args.seed)
    if seed_id is None:
        raise SystemExit(f'Seed title {args.seed!r} not found as a namespace-0 non-redirect page')

    print(f'BFS from {args.seed!r} (page_id={seed_id}) to {args.target} nodes...')
    visited = {seed_id}
    queue = deque([seed_id])
    while queue and len(visited) < args.target:
        node = queue.popleft()
        for nxt in sorted(adjacency.get(node, ())):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
                if len(visited) >= args.target:
                    break
    print(f'  visited {len(visited):,} nodes')

    edges = set()
    for u in visited:
        for v in adjacency.get(u, ()):
            if v in visited and v != u:
                edges.add(tuple(sorted((u, v))))
    print(f'  {len(edges):,} real undirected edges among visited nodes')

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'titles.json').write_text(
        json.dumps({pid: id_to_title[pid] for pid in visited}, indent=2, ensure_ascii=False))
    with open(out_dir / 'positive_edges.csv', 'w') as f:
        f.write('id1,id2\n')
        for a, b in edges:
            f.write(f'{a},{b}\n')
    print(f'Wrote {out_dir}/titles.json and {out_dir}/positive_edges.csv')


if __name__ == '__main__':
    main()
