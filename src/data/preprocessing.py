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
    if pd.isna(text):
        return None, None

    infobox_matches = re.findall(r"{{(.*?)}}", text, re.DOTALL)

    if len(infobox_matches) == 0:
        return None, Description(text)

    if len(infobox_matches) > 1:
        infobox_matches = list(dict.fromkeys(infobox_matches))
        print(f'Found {len(infobox_matches)} info-boxes in text')

    infobox_match: str = infobox_matches[0]

    infobox = Infobox(infobox_match)

    text = re.subn(r"{{(.*?)}}", '', text, re.DOTALL)[0]

    description = Description(text)

    return infobox, description
