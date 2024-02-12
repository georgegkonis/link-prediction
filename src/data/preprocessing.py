import re

import pandas as pd

from src.data.description import Description
from src.data.infobox import Infobox


def remove_extra_spaces(text: str) -> str | None:
    if pd.isna(text):
        return None

    return re.sub(r'\s+', ' ', text)


def remove_html_comments(text: str) -> str | None:
    if pd.isna(text):
        return None

    text: str = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    if str.isspace(text):
        return None

    return text


def merge_infoboxes(infoboxes: list[Infobox]) -> Infobox:
    if len(infoboxes) == 0:
        return Infobox('')

    if len(infoboxes) == 1:
        return infoboxes[0]

    merged_infobox = Infobox('')

    for infobox in infoboxes:
        merged_infobox.title = infobox.title if infobox.title else merged_infobox.title
        merged_infobox.content = {**merged_infobox.content, **infobox.content}
        merged_infobox.other = [*merged_infobox.other, *infobox.other]

    return merged_infobox


def extract_infobox(text: str) -> tuple[Infobox | None, Description | None]:
    # If there is no text, return empty dict and empty string
    if pd.isna(text):
        return None, None

    # Search for all info-boxes in the text
    infobox_matches = re.findall(r"{{(.*?)}}", text, re.DOTALL)

    if len(infobox_matches) == 0:
        return None, Description(text)

    if len(infobox_matches) > 1:
        infobox_matches = list(dict.fromkeys(infobox_matches))
        print(f'Found {len(infobox_matches)} info-boxes in text')

    infobox_match: str = infobox_matches[0]

    # Parse key-value pairs from infobox
    infobox = Infobox(infobox_match)

    # Extract description
    text = re.subn(r"{{(.*?)}}", '', text, re.DOTALL)[0]

    description = Description(text)

    return infobox, description


# Load 1000 random records from the nodes
df = pd.read_csv('./data/nodes.tsv', sep='\t', nrows=1000, index_col=0)

df['text'] = df['text'].apply(remove_extra_spaces)
df['text'] = df['text'].apply(remove_html_comments)


# Function to extract fields from Infobox and Description objects
def extract_fields(text):
    infobox, description = extract_infobox(text)

    # Extracting fields from the infobox and description
    infobox_title = infobox.title if infobox else None
    infobox_content = infobox.content if infobox else {}
    infobox_other = infobox.other if infobox else []
    description_text = description.text if description else ""
    description_keywords = description.keywords if description else set()

    return infobox_title, infobox_content, infobox_other, description_text, description_keywords


# Apply the function and create new columns
df[['infobox_title', 'infobox_content', 'infobox_other', 'description_text', 'description_keywords']] = \
    (pd.DataFrame(df['text'].apply(extract_fields).tolist(), index=df.index))

# %%
# Save the preprocessed dataframe
df.to_csv('./data/pp_nodes.csv', index=False)
