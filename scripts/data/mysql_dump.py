"""Minimal streaming parser for MediaWiki/MySQL dump INSERT statements."""
import gzip
import re

_TUPLE_RE = re.compile(r"\((?:[^()'\\]|\\.|'(?:[^'\\]|\\.)*')*\)")


def _split_row(tuple_str: str) -> list:
    """Split one `(a,b,'c,d')`-style tuple into fields, unescaping SQL strings."""
    fields = []
    i, n = 0, len(tuple_str)
    buf = []
    in_str = False
    j = 1  # skip leading '('
    while j < n - 1:
        c = tuple_str[j]
        if in_str:
            if c == '\\' and j + 1 < n:
                nxt = tuple_str[j + 1]
                unescape = {'\\': '\\', "'": "'", 'n': '\n', 't': '\t', '0': '\0', 'r': '\r'}
                buf.append(unescape.get(nxt, nxt))
                j += 2
                continue
            if c == "'":
                in_str = False
                j += 1
                continue
            buf.append(c)
            j += 1
        else:
            if c == "'":
                in_str = True
                j += 1
                continue
            if c == ',':
                fields.append(''.join(buf))
                buf = []
                j += 1
                continue
            buf.append(c)
            j += 1
    fields.append(''.join(buf))
    return fields


def iter_insert_rows(path: str, table: str):
    """Yield each row (list of string fields, NULL -> None) from `INSERT INTO \`table\` VALUES (...);` lines."""
    opener = gzip.open if path.endswith('.gz') else open
    prefix = f"INSERT INTO `{table}` VALUES "
    with opener(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.startswith(prefix):
                continue
            body = line[len(prefix):]
            for m in _TUPLE_RE.finditer(body):
                fields = _split_row(m.group(0))
                yield [None if v == 'NULL' else v for v in fields]
