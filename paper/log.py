"""Append one row to paper/experiments.csv.
Usage: python paper/log.py --id E032 --cat blend --desc "..." --base 0.4915 --value 0.5238 --folds 5/5
       [--metric oof_bacc] [--ci "0.01,0.05"] [--config "w=0.5"] [--files "experiments/x.py;out/x.log"] [--notes "..."]
Also importable: from paper.log import log_row"""
import argparse
import csv
import datetime as dt
import pathlib

LOG = pathlib.Path(__file__).resolve().parent / 'experiments.csv'
FIELDS = ['date', 'id', 'category', 'description', 'config', 'metric', 'baseline', 'value', 'delta', 'folds_pos',
          'ci95', 'files', 'notes']


def log_row(**kw):
    new = not LOG.exists()
    row = {k: kw.get(k, '') for k in FIELDS}
    row['date'] = row['date'] or dt.date.today().isoformat()
    if row['baseline'] not in ('', None) and row['value'] not in ('', None):
        try:
            row['delta'] = f"{float(row['value']) - float(row['baseline']):+.4f}"
        except ValueError:
            pass
    with open(LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


ACT = LOG.parent / 'activity_log.csv'
ACT_FIELDS = ['timestamp_utc', 'actor', 'action', 'target', 'details']


def log_activity(action, target='', details='', actor='claude-code', ts=''):
    """Append one row to paper/activity_log.csv: every action that changes state outside a pure computation
    (Kaggle pushes / submissions / dataset creation, downloads, generation runs, git pushes, file deliveries)."""
    new = not ACT.exists()
    row = dict(timestamp_utc=ts or dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M'), actor=actor,
               action=action, target=target, details=details)
    with open(ACT, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ACT_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k in ('id', 'cat', 'desc', 'config', 'metric', 'base', 'value', 'folds', 'ci', 'files', 'notes', 'date',
              'act', 'target', 'details', 'actor'):
        ap.add_argument(f'--{k}', default='')
    a = ap.parse_args()
    if a.act:   # activity log: python paper/log.py --act "kaggle push" --target ntl-v6d --details "test run"
        print(log_activity(a.act, a.target, a.details, a.actor or 'claude-code'))
        raise SystemExit
    print(log_row(id=a.id, category=a.cat, description=a.desc, config=a.config, metric=a.metric or 'oof_bacc',
                  baseline=a.base, value=a.value, folds_pos=a.folds, ci95=a.ci, files=a.files, notes=a.notes,
                  date=a.date))
