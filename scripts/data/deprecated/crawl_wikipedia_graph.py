"""
SUPERSEDED — kept for reference only. The live MediaWiki API hard-rate-limits
this environment's IP (sustained 429s with ~45s Retry-After on every request,
not just a burst limit), making any useful crawl scale impractical here.
Use build_from_wikidump.py instead: it gets the same real link data in bulk
from Wikipedia's official SQL dumps (dumps.wikimedia.org), which is not
rate-limited the same way and is dramatically faster (minutes vs. hours).

Crawl a real, connected subgraph of English Wikipedia via the live MediaWiki
API, starting from a seed article and following genuine outgoing links
(breadth-first), up to a target node count. Unlike the DSAA 2023 dataset,
every edge here is a real hyperlink and every node's text is a real article
extract — there is no negative-sampling artifact to correct after the fact,
because we build the labels ourselves in build_wikipedia_cs_8k_dataset.py from
these genuine edges.

Writes:
    <output>/nodes.tsv          — id, text  (same schema as data/raw/dsaa/nodes.tsv)
    <output>/positive_edges.csv — id1, id2  (real hyperlinks, undirected, deduped)
    <output>/titles.json        — id -> title mapping, for readable examples later

Usage:
    python -m scripts.data.crawl_wikipedia_graph --seed "Computer science" --target 8000
"""
import argparse
import json
import pathlib
import time
from collections import deque

import pandas as pd
import requests

API = 'https://en.wikipedia.org/w/api.php'
HEADERS = {'User-Agent': 'link-prediction-thesis-crawler/1.0 (CEID diploma thesis; contact via GitHub)'}
BATCH = 50
MIN_INTERVAL = 1.2  # seconds between requests — this sandbox's IP appears to be shared/rate-limited
_last_call = [0.0]


def _throttle():
    wait = _last_call[0] + MIN_INTERVAL - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def api_get(params: dict) -> dict:
    params = {**params, 'format': 'json', 'action': 'query'}
    last_err = None
    for attempt in range(8):
        _throttle()
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=20)
            if r.status_code == 429:
                retry_after = float(r.headers.get('Retry-After', 15))
                print(f'  [429, attempt {attempt+1}/8] backing off {retry_after:.0f}s', flush=True)
                time.sleep(retry_after)
                _last_call[0] = time.time()
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            print(f'  [retry {attempt+1}/8] {e}', flush=True)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f'API request failed after retries ({last_err}): {params}')


def fetch_links(titles: list[str]) -> dict[str, list[str]]:
    """Real outgoing links (main namespace only) for a batch of titles."""
    out = {t: [] for t in titles}
    params = {
        'prop': 'links', 'titles': '|'.join(titles),
        'plnamespace': 0, 'pllimit': 'max', 'redirects': 1,
    }
    seen_pages = set()
    while True:
        data = api_get(params)
        pages = data.get('query', {}).get('pages', {})
        for page in pages.values():
            title = page.get('title')
            if title is None:
                continue
            links = [l['title'] for l in page.get('links', [])]
            out.setdefault(title, [])
            out[title].extend(links)
            seen_pages.add(title)
        cont = data.get('continue')
        if not cont:
            break
        params.update(cont)
    return out


def fetch_extracts(titles: list[str]) -> dict[str, str]:
    """Plain-text extracts for a batch of titles."""
    out = {}
    params = {
        'prop': 'extracts', 'titles': '|'.join(titles),
        'explaintext': 1, 'exlimit': 'max', 'redirects': 1,
    }
    data = api_get(params)
    pages = data.get('query', {}).get('pages', {})
    for page in pages.values():
        title = page.get('title')
        extract = page.get('extract', '')
        if title:
            out[title] = extract
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', default='Computer science')
    parser.add_argument('--target', type=int, default=8000)
    parser.add_argument('--output', default='data/raw/wikipedia_cs_8k')
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    visited_links: dict[str, list[str]] = {}
    queued = {args.seed}
    queue = deque([args.seed])
    t0 = time.time()

    while queue and len(visited_links) < args.target:
        batch = []
        while queue and len(batch) < BATCH:
            batch.append(queue.popleft())
        links_batch = fetch_links(batch)
        for title, links in links_batch.items():
            visited_links[title] = links
            if len(visited_links) >= args.target:
                break
            for link in links:
                if link not in queued:
                    queued.add(link)
                    queue.append(link)
        if len(visited_links) % 500 < BATCH:
            print(f'  visited={len(visited_links)}  queued={len(queue)}  '
                  f'elapsed={time.time()-t0:.0f}s', flush=True)

    node_set = set(visited_links.keys())
    print(f'Crawled {len(node_set)} nodes in {time.time()-t0:.0f}s')

    # ---- Real positive edges: keep only links landing inside our final node set ----
    edges = set()
    for src, links in visited_links.items():
        for dst in links:
            if dst in node_set and dst != src:
                edges.add(tuple(sorted((src, dst))))
    print(f'Real edges among crawled nodes: {len(edges)}')

    title_to_id = {t: i for i, t in enumerate(sorted(node_set))}

    # ---- Fetch extracts in batches ----
    titles_list = list(node_set)
    texts = {}
    for i in range(0, len(titles_list), 20):
        batch = titles_list[i:i + 20]
        texts.update(fetch_extracts(batch))
        if i % 500 < 20:
            print(f'  extracts fetched={i+len(batch)}/{len(titles_list)}  '
                  f'elapsed={time.time()-t0:.0f}s', flush=True)

    nodes_df = pd.DataFrame({
        'id': [title_to_id[t] for t in titles_list],
        'text': [texts.get(t, '') for t in titles_list],
    }).set_index('id').sort_index()
    nodes_df.to_csv(out_dir / 'nodes.tsv', sep='\t')

    edges_df = pd.DataFrame(
        [(title_to_id[a], title_to_id[b]) for a, b in edges], columns=['id1', 'id2'])
    edges_df.to_csv(out_dir / 'positive_edges.csv', index=False)

    (out_dir / 'titles.json').write_text(json.dumps(
        {v: k for k, v in title_to_id.items()}, indent=2, ensure_ascii=False))

    n_empty = (nodes_df['text'].str.len() == 0).sum()
    print(f'Done. Nodes: {len(nodes_df)}  Edges: {len(edges_df)}  '
          f'Nodes with empty extract: {n_empty}  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
