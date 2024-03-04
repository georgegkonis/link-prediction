"""
Preprocess raw Wikipedia nodes: clean text, extract infoboxes and descriptions.

Input:  data/raw/nodes.tsv
Output: data/interim/pp_nodes.csv
"""

import argparse

import pandas as pd

from src.data.preprocessing import extract_infobox, remove_extra_spaces, remove_html_comments


def extract_fields(text):
    infobox, description = extract_infobox(text)
    return (
        infobox.title if infobox else None,
        infobox.content if infobox else {},
        infobox.other if infobox else [],
        description.text if description else '',
        description.keywords if description else set(),
    )


def main(input_path: str, output_path: str, nrows: int | None = None):
    df = pd.read_csv(input_path, sep='\t', nrows=nrows, index_col=0)

    df['text'] = df['text'].apply(remove_extra_spaces)
    df['text'] = df['text'].apply(remove_html_comments)

    df[['infobox_title', 'infobox_content', 'infobox_other', 'description_text', 'description_keywords']] = (
        pd.DataFrame(df['text'].apply(extract_fields).tolist(), index=df.index)
    )

    df.to_csv(output_path, index=False)
    print(f'Saved {len(df)} records to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw/nodes.tsv')
    parser.add_argument('--output', default='data/interim/pp_nodes.csv')
    parser.add_argument('--nrows', type=int, default=None)
    args = parser.parse_args()

    main(args.input, args.output, args.nrows)
