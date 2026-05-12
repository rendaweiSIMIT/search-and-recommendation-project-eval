"""Eval-side entry point for exp/eda branch.

Replaces the baseline ``infer.py`` (which builds a PCVRHyFormer model and
runs real predictions). On the EDA branch we:

  1. Stream every parquet under ``$EVAL_DATA_PATH`` and accumulate the same
     statistics ``eda.py`` collects on the training side, minus the
     label-dependent ones (eval data has hidden labels). Dumps:

       - ``$EVAL_RESULT_PATH/eda_report_eval.json``
       - log lines on stdout (captured by the platform Logs panel)

  2. Write a placeholder ``predictions.json`` with constant probability for
     every ``user_id`` seen, in the exact ``{"predictions": {uid: prob}}``
     schema the baseline infer.py produces. This keeps the platform's
     eval pipeline happy. The AUC of this branch is intentionally garbage --
     we are running it to read the EDA log, not to score.

No model is loaded, no checkpoint is required. The platform's
``MODEL_OUTPUT_PATH`` env var is read but unused.

Environment variables expected (same as baseline infer.py):
    EVAL_DATA_PATH    Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH  Directory for the generated predictions.json + eda JSON.
"""
import json
import logging
import os
import sys
import time
from typing import List

import numpy as np
import pyarrow.parquet as pq

from eda import setup_logger, stream_eda


CONSTANT_PROB = 0.1   # ~ matches global positive rate, irrelevant but plausible


def collect_user_ids(data_dir: str) -> List[int]:
    """Stream every parquet under ``data_dir`` and return user_id in order."""
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No .parquet files in {data_dir}')
    logging.info(f'collect_user_ids: scanning {len(files)} files')
    out: List[int] = []
    t0 = time.time()
    for fi, fpath in enumerate(files):
        try:
            pf = pq.ParquetFile(fpath)
        except Exception as exc:
            logging.warning(f'Failed to open {fpath}: {exc}; skipping')
            continue
        if 'user_id' not in pf.schema_arrow.names:
            logging.warning(f'No user_id column in {fpath}; skipping')
            continue
        for rg_idx in range(pf.metadata.num_row_groups):
            try:
                batch = pf.read_row_group(rg_idx, columns=['user_id'])
            except Exception as exc:
                logging.warning(
                    f'Failed to read RG {rg_idx} from {fpath}: {exc}; skipping')
                continue
            uids = batch.column('user_id').to_numpy(zero_copy_only=False)
            out.extend(int(u) for u in uids)
        if (fi + 1) % 100 == 0 or fi == len(files) - 1:
            logging.info(
                f'  collect_user_ids progress: {fi + 1}/{len(files)} files, '
                f'{len(out):,} rows, {time.time() - t0:.0f}s')
    return out


def write_dummy_predictions(result_dir: str, user_ids: List[int]) -> str:
    """Write a placeholder predictions.json keyed by user_id."""
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, 'predictions.json')
    predictions = {
        'predictions': {str(uid): CONSTANT_PROB for uid in user_ids},
    }
    with open(path, 'w') as f:
        json.dump(predictions, f)
    logging.info(
        f'Wrote {len(user_ids):,} placeholder predictions to {path} '
        f'(constant prob={CONSTANT_PROB})')
    return path


def main() -> None:
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')
    if not data_dir:
        raise ValueError('EVAL_DATA_PATH not set')
    if not result_dir:
        raise ValueError('EVAL_RESULT_PATH not set')

    setup_logger(result_dir)
    logging.info('=== exp/eda eval-side entry point ===')
    logging.info(f'EVAL_DATA_PATH={data_dir}')
    logging.info(f'EVAL_RESULT_PATH={result_dir}')
    logging.info(f'(MODEL_OUTPUT_PATH={os.environ.get("MODEL_OUTPUT_PATH")!r} '
                 f'-- ignored on this branch)')

    # 1. EDA pass on the test data (labels are hidden, so no 1-D AUC).
    eda_output = os.path.join(result_dir, 'eda_report_eval.json')
    logging.info('--- Step 1/2: streaming EDA on eval parquets ---')
    stream_eda(
        data_dir=data_dir,
        output_path=eda_output,
        log_dir=result_dir,
        is_training=False,
    )

    # 2. Collect user_ids for predictions.json. Done in a separate pass so
    #    that step 1's per-feature accumulators stay focused; the EVA pass
    #    is read-once already so the extra scan is cheap.
    logging.info('--- Step 2/2: collecting user_ids for placeholder predictions ---')
    user_ids = collect_user_ids(data_dir)
    write_dummy_predictions(result_dir, user_ids)

    logging.info('=== eval-side EDA + placeholder predictions DONE ===')


if __name__ == '__main__':
    main()
