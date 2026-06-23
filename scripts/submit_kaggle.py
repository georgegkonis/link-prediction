"""
Submit a predictions CSV to the DSAA 2023 Kaggle competition and poll for
scored results, using `kagglesdk` (the library backing Kaggle's current
kaggle-cli, github.com/Kaggle/kaggle-cli) authenticated via KAGGLE_API_TOKEN.

KAGGLE_API_TOKEN in .env (already used by scripts/download_data.py /
kagglehub) is sufficient -- no separate username/key pair needed.

Usage:
    python -m scripts.submit_kaggle --file outputs/predictions/cascade_submission.csv \
        --message "CascadeLP val macro-F1 0.9986"
    python -m scripts.submit_kaggle --check   # list recent submissions and scores
"""

import argparse
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

COMPETITION = 'dsaa-2023-competition'

if not os.environ.get('KAGGLE_API_TOKEN'):
    sys.exit('KAGGLE_API_TOKEN not set in .env. See kaggle.com/settings > API.')

from kagglesdk import KaggleClient  # noqa: E402
from kagglesdk.competitions.types.competition_api_service import (  # noqa: E402
    ApiCreateSubmissionRequest,
    ApiListSubmissionsRequest,
    ApiStartSubmissionUploadRequest,
)


def submit(file_path: str, message: str):
    client = KaggleClient()
    api = client.competitions.competition_api_client

    start_req = ApiStartSubmissionUploadRequest()
    start_req.competition_name = COMPETITION
    start_req.content_length = os.path.getsize(file_path)
    start_req.last_modified_epoch_seconds = int(os.path.getmtime(file_path))
    start_req.file_name = os.path.basename(file_path)
    start_resp = api.start_submission_upload(start_req)

    print(f'Uploading {file_path}...')
    with open(file_path, 'rb') as f:
        resp = requests.put(start_resp.create_url, data=f)
    if resp.status_code not in (200, 201):
        sys.exit(f'Upload failed: HTTP {resp.status_code} {resp.text[:300]}')

    create_req = ApiCreateSubmissionRequest()
    create_req.competition_name = COMPETITION
    create_req.blob_file_tokens = start_resp.token
    create_req.submission_description = message
    api.create_submission(create_req)
    print('Submitted. Use --check to poll the score once it finishes grading.')


def check(wait: bool = False):
    client = KaggleClient()
    api = client.competitions.competition_api_client
    req = ApiListSubmissionsRequest()
    req.competition_name = COMPETITION

    while True:
        subs = api.list_submissions(req).submissions
        if not subs:
            print('No submissions found.')
            return
        latest = subs[0]
        if wait and latest.status.name == 'PENDING':
            print('Pending... waiting 15s')
            time.sleep(15)
            continue
        break

    print(f'{"date":<26} {"status":<12} {"public_score":<14} {"description"}')
    for s in subs[:10]:
        score = s.public_score if s.public_score else '-'
        print(f'{str(s.date):<26} {s.status.name:<12} {str(score):<14} {s.description}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--file', help='Path to a submission CSV (id,label)')
    p.add_argument('--message', default='', help='Submission message/description')
    p.add_argument('--check', action='store_true', help='List recent submissions and scores instead of submitting')
    p.add_argument('--wait', action='store_true', help='With --check, poll until the latest submission is scored')
    args = p.parse_args()

    if args.check:
        check(wait=args.wait)
    elif args.file:
        submit(args.file, args.message)
    else:
        p.error('pass --file to submit or --check to poll scores')
