"""
Submit predictions to Kaggle and log results.

Writes:
    outputs/predictions/dsaa/kaggle_scores.csv

Usage:
    python -m scripts.analysis.submit_dsaa_kaggle --file outputs/predictions/dsaa/cascade_submission.csv
"""

import argparse
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

import pathlib
from omegaconf import OmegaConf

from src.utils.log_utils import setup_logging

cfg = OmegaConf.load(pathlib.Path(__file__).parent.parent / 'configs' / 'config.yaml')
log = setup_logging('submit_kaggle')
COMPETITION = cfg.kaggle.competition
LOG_PATH = cfg.paths.log_path
LOG_COLUMNS = ['ref', 'date', 'file_name', 'description', 'status', 'public_score', 'private_score']

if not os.environ.get('KAGGLE_API_TOKEN'):
    sys.exit('KAGGLE_API_TOKEN not set in .env. See kaggle.com/settings > API.')

from kagglesdk import KaggleClient  # noqa: E402
from kagglesdk.competitions.types.competition_api_service import (  # noqa: E402
    ApiCreateSubmissionRequest,
    ApiListSubmissionsRequest,
    ApiStartSubmissionUploadRequest,
)


def _log_upsert(subs) -> pd.DataFrame:
    rows = {
        str(s.ref): {
            'ref': str(s.ref), 'date': str(s.date), 'file_name': s.file_name,
            'description': s.description, 'status': s.status.name,
            'public_score': s.public_score or '', 'private_score': s.private_score or '',
        }
        for s in subs
    }
    if os.path.exists(LOG_PATH):
        existing = pd.read_csv(LOG_PATH, dtype=str).set_index('ref').to_dict('index')
        existing.update(rows)
        rows = existing
    log = pd.DataFrame(rows.values(), columns=LOG_COLUMNS)
    log = log.sort_values('date', ascending=False)
    log.to_csv(LOG_PATH, index=False)
    return log


def _list(api) -> list:
    req = ApiListSubmissionsRequest()
    req.competition_name = COMPETITION
    return api.list_submissions(req).submissions


def submit(file_path: str, message: str, wait: bool):
    client = KaggleClient()
    api = client.competitions.competition_api_client
    file_name = os.path.basename(file_path)

    start_req = ApiStartSubmissionUploadRequest()
    start_req.competition_name = COMPETITION
    start_req.content_length = os.path.getsize(file_path)
    start_req.last_modified_epoch_seconds = int(os.path.getmtime(file_path))
    start_req.file_name = file_name
    start_resp = api.start_submission_upload(start_req)

    log.info('Uploading %s...', file_path)
    with open(file_path, 'rb') as f:
        resp = requests.put(start_resp.create_url, data=f)
    if resp.status_code not in (200, 201):
        sys.exit(f'Upload failed: HTTP {resp.status_code} {resp.text[:300]}')

    create_req = ApiCreateSubmissionRequest()
    create_req.competition_name = COMPETITION
    create_req.blob_file_tokens = start_resp.token
    create_req.submission_description = message
    api.create_submission(create_req)
    log.info('Submitted.')

    # The newest submission matching this file_name is ours (Kaggle assigns
    # the `ref` on creation; list_submissions is our only way to learn it).
    while True:
        subs = _list(api)
        ours = next((s for s in subs if s.file_name == file_name), None)
        if ours is None:
            time.sleep(3)
            continue
        if wait and ours.status.name == 'PENDING':
            log.info('Pending... waiting 15s')
            time.sleep(15)
            continue
        break

    _log_upsert(subs)
    score = ours.public_score or '(pending)'
    log.info('ref=%s  status=%s  public_score=%s', ours.ref, ours.status.name, score)
    log.info('Logged → %s', LOG_PATH)


def check(wait: bool = False):
    client = KaggleClient()
    api = client.competitions.competition_api_client

    while True:
        subs = _list(api)
        if not subs:
            log.info('No submissions found.')
            return
        if wait and subs[0].status.name == 'PENDING':
            log.info('Pending... waiting 15s')
            time.sleep(15)
            continue
        break

    scores = _log_upsert(subs)
    log.info('\n%s', scores.head(10).to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--file', help='Path to a submission CSV (id,label)')
    p.add_argument('--message', default='', help='Submission message/description')
    p.add_argument('--check', action='store_true', help='List recent submissions and scores instead of submitting')
    p.add_argument('--wait', action='store_true', help='Poll until the relevant submission is scored')
    args = p.parse_args()

    if args.check:
        check(wait=args.wait)
    elif args.file:
        submit(args.file, args.message, wait=args.wait)
    else:
        p.error('pass --file to submit or --check to poll scores')
