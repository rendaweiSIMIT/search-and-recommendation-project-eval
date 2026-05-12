"""TAAC2026 PCVR EDA script.

Streams every parquet file under TRAIN_DATA_PATH, accumulates per-feature
statistics (null rate, value range, unique cardinality, 1-D AUC vs label,
sequence length distribution, timestamp-derived hour/dow distribution,
``label_time - timestamp`` conversion-delay distribution), and dumps:

  - ``$TRAIN_CKPT_PATH/eda_report.json``     -- full machine-readable dump
  - stdout + ``$TRAIN_LOG_PATH/eda.log``     -- human-readable summary

Designed to replace ``train.py`` as the run.sh entry point on the
``exp/eda`` branch: no model is built, no checkpoint is produced. The
sole goal is to surface insights that inform feature-engineering or
architectural decisions on follow-up branches.

Reused by the eval-side ``infer.py`` shim with ``--mode eval`` to dump
the same statistics for test data (no labels available so 1-D AUC is
skipped on that side).
"""
import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# sklearn is imported lazily so a missing install only breaks 1-D AUC
# while structural stats still flow.
try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


# ─────────────────────────── Logging setup ──────────────────────────────────

def setup_logger(log_dir: Optional[str]) -> logging.Logger:
    """Configure root logger to write to stdout (always) and an optional file."""
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%m/%d/%y %H:%M:%S')

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, 'eda.log'), 'w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ────────────────────────── Sampling caps ────────────────────────────────────

UNIQUE_CAP = 50_000          # max unique values to track per feature
AUC_SAMPLE_CAP = 200_000     # max (value, label) pairs per feature for 1-D AUC
LEN_SAMPLE_CAP = 50_000      # max sequence-length samples
DELAY_HIST_BINS = np.array(  # buckets for label_time - timestamp distribution
    [1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600, 7200, 86400], dtype=np.int64)


# ────────────────────────── Per-feature accumulators ─────────────────────────

class ScalarAccum:
    """Streaming stats for a scalar int / float column."""

    def __init__(self, name: str, dtype_kind: str) -> None:
        self.name = name
        self.dtype_kind = dtype_kind  # 'i'|'u'|'f'
        self.n_total = 0
        self.n_null = 0
        self.n_zero = 0
        self.minv = float('inf')
        self.maxv = float('-inf')
        self.sumv = 0.0
        self.sumsq = 0.0
        self.unique: set = set()
        self._auc_v: List[float] = []
        self._auc_l: List[int] = []

    def update(self, values: np.ndarray, labels: Optional[np.ndarray]) -> None:
        n = len(values)
        if n == 0:
            return
        self.n_total += n
        if self.dtype_kind == 'f':
            null_mask = np.isnan(values)
            self.n_null += int(null_mask.sum())
            non_null = values[~null_mask]
        else:
            # int columns: null already mapped to 0 upstream; we count zeros
            # but don't separately count nulls (they overlap).
            non_null = values
        self.n_zero += int((values == 0).sum())
        if len(non_null) > 0:
            self.minv = min(self.minv, float(non_null.min()))
            self.maxv = max(self.maxv, float(non_null.max()))
            f = non_null.astype(np.float64)
            self.sumv += float(f.sum())
            self.sumsq += float((f * f).sum())
            if len(self.unique) < UNIQUE_CAP:
                space = UNIQUE_CAP - len(self.unique)
                if self.dtype_kind in 'iu':
                    self.unique.update(non_null.astype(np.int64).tolist()[:space])
                else:
                    self.unique.update(np.round(non_null, 4).tolist()[:space])
        if labels is not None and len(self._auc_v) < AUC_SAMPLE_CAP:
            space = AUC_SAMPLE_CAP - len(self._auc_v)
            take = min(n, space)
            self._auc_v.extend(values[:take].astype(np.float64).tolist())
            self._auc_l.extend(labels[:take].astype(np.int64).tolist())

    def finalize(self) -> Dict[str, Any]:
        n_valid = max(1, self.n_total - self.n_null)
        mean = self.sumv / n_valid
        var = self.sumsq / n_valid - mean * mean
        std = max(0.0, var) ** 0.5
        r: Dict[str, Any] = {
            'type': 'scalar',
            'n_total': self.n_total,
            'null_rate': self.n_null / max(1, self.n_total),
            'zero_rate': self.n_zero / max(1, self.n_total),
            'min': None if self.minv == float('inf') else self.minv,
            'max': None if self.maxv == float('-inf') else self.maxv,
            'mean': mean,
            'std': std,
            'unique_sampled': len(self.unique),
            'unique_capped': len(self.unique) >= UNIQUE_CAP,
        }
        if _HAS_SKLEARN and self._auc_v:
            try:
                v = np.array(self._auc_v)
                l = np.array(self._auc_l)
                if len(np.unique(l)) >= 2 and len(np.unique(v)) >= 2:
                    auc = float(roc_auc_score(l, v))
                    r['auc_1d'] = auc
                    r['auc_signal'] = abs(auc - 0.5) * 2  # 0=no signal, 1=perfect
            except Exception as exc:  # pragma: no cover
                r['auc_error'] = str(exc)
        return r


class ArrayAccum:
    """Streaming stats for a list<int> / list<float> column."""

    def __init__(self, name: str, dtype_kind: str) -> None:
        self.name = name
        self.dtype_kind = dtype_kind
        self.n_total = 0
        self.n_empty = 0
        self.lens: List[int] = []
        self.flat_zero = 0
        self.flat_count = 0
        self.minv = float('inf')
        self.maxv = float('-inf')
        self.unique: set = set()
        self._auc_mean_v: List[float] = []
        self._auc_first_v: List[float] = []
        self._auc_max_v: List[float] = []
        self._auc_l: List[int] = []

    def update(
        self,
        offsets: np.ndarray,
        flat_values: np.ndarray,
        n_rows: int,
        labels: Optional[np.ndarray],
    ) -> None:
        """offsets has shape (n_rows+1,) and flat_values are the concatenated lists."""
        self.n_total += n_rows
        # Per-row iteration; pyarrow already gave us a flat buffer so this is cheap.
        for i in range(n_rows):
            s = int(offsets[i])
            e = int(offsets[i + 1])
            ln = e - s
            if ln == 0:
                self.n_empty += 1
                if labels is not None and len(self._auc_mean_v) < AUC_SAMPLE_CAP:
                    self._auc_mean_v.append(0.0)
                    self._auc_first_v.append(0.0)
                    self._auc_max_v.append(0.0)
                    self._auc_l.append(int(labels[i]))
                continue
            arr = flat_values[s:e]
            if len(self.lens) < LEN_SAMPLE_CAP:
                self.lens.append(ln)
            # flat-value stats (avoid full Python iteration -- use numpy reductions)
            self.flat_count += ln
            self.flat_zero += int((arr == 0).sum())
            a_min = float(arr.min())
            a_max = float(arr.max())
            self.minv = min(self.minv, a_min)
            self.maxv = max(self.maxv, a_max)
            # unique sampling: capped, only take a few values per row to limit cost
            if len(self.unique) < UNIQUE_CAP:
                space = UNIQUE_CAP - len(self.unique)
                # First 8 values per row -- enough to estimate cardinality
                sample = arr[:8]
                if self.dtype_kind in 'iu':
                    self.unique.update(sample.astype(np.int64).tolist()[:space])
                else:
                    self.unique.update(np.round(sample, 4).tolist()[:space])
            # 1-D AUC via summary stats
            if labels is not None and len(self._auc_mean_v) < AUC_SAMPLE_CAP:
                self._auc_mean_v.append(float(arr.mean()))
                nz = arr[arr != 0]
                self._auc_first_v.append(float(nz[0]) if len(nz) > 0 else 0.0)
                self._auc_max_v.append(a_max)
                self._auc_l.append(int(labels[i]))

    def finalize(self) -> Dict[str, Any]:
        len_arr = np.array(self.lens) if self.lens else np.array([0])
        r: Dict[str, Any] = {
            'type': 'array',
            'n_total': self.n_total,
            'empty_rate': self.n_empty / max(1, self.n_total),
            'len_min': int(len_arr.min()),
            'len_max': int(len_arr.max()),
            'len_mean': float(len_arr.mean()),
            'len_p50': float(np.median(len_arr)),
            'len_p95': float(np.percentile(len_arr, 95)),
            'value_min': None if self.minv == float('inf') else self.minv,
            'value_max': None if self.maxv == float('-inf') else self.maxv,
            'flat_zero_rate': self.flat_zero / max(1, self.flat_count),
            'unique_sampled': len(self.unique),
            'unique_capped': len(self.unique) >= UNIQUE_CAP,
        }
        if _HAS_SKLEARN and self._auc_l:
            try:
                l = np.array(self._auc_l)
                if len(np.unique(l)) >= 2:
                    for tag, buf in [
                        ('auc_via_mean', self._auc_mean_v),
                        ('auc_via_first_nz', self._auc_first_v),
                        ('auc_via_max', self._auc_max_v),
                    ]:
                        v = np.array(buf)
                        if len(np.unique(v)) >= 2:
                            r[tag] = float(roc_auc_score(l, v))
                    # Best of the three as the "feature strength"
                    aucs = [r[k] for k in ('auc_via_mean', 'auc_via_first_nz',
                                           'auc_via_max') if k in r]
                    if aucs:
                        best = max(aucs, key=lambda a: abs(a - 0.5))
                        r['auc_best'] = best
                        r['auc_signal'] = abs(best - 0.5) * 2
            except Exception as exc:  # pragma: no cover
                r['auc_error'] = str(exc)
        return r


# ─────────────────────────── Sample-level accumulators ───────────────────────

class SampleAccum:
    """Top-level stats: label distribution, timestamp distribution, delays."""

    def __init__(self) -> None:
        self.n = 0
        self.label_counts: Dict[int, int] = defaultdict(int)
        self.ts_min = None
        self.ts_max = None
        self.hour_counts = np.zeros(24, dtype=np.int64)
        self.dow_counts = np.zeros(7, dtype=np.int64)
        # label_time - timestamp delay histogram (positive samples only and overall)
        self.delay_hist_all = np.zeros(len(DELAY_HIST_BINS) + 1, dtype=np.int64)
        self.delay_hist_pos = np.zeros(len(DELAY_HIST_BINS) + 1, dtype=np.int64)
        self.delay_sum = 0.0
        self.delay_count = 0
        self.delay_neg_count = 0    # label_time < timestamp (unexpected)
        # hour x label cross
        self.hour_x_pos = np.zeros(24, dtype=np.int64)

    def update(
        self,
        timestamps: np.ndarray,
        label_types: Optional[np.ndarray],
        label_times: Optional[np.ndarray],
    ) -> None:
        n = len(timestamps)
        self.n += n
        if self.ts_min is None or timestamps.min() < self.ts_min:
            self.ts_min = int(timestamps.min())
        if self.ts_max is None or timestamps.max() > self.ts_max:
            self.ts_max = int(timestamps.max())
        hours = ((timestamps // 3600) % 24).astype(np.int64)
        np.add.at(self.hour_counts, hours, 1)
        # day-of-week (1970-01-01 was a Thursday=3)
        days = timestamps // 86400
        dow = ((days + 3) % 7).astype(np.int64)
        np.add.at(self.dow_counts, dow, 1)
        if label_types is not None:
            for lt in np.unique(label_types):
                self.label_counts[int(lt)] += int((label_types == lt).sum())
            pos_mask = (label_types == 2)
            np.add.at(self.hour_x_pos, hours[pos_mask], 1)
        if label_times is not None and label_types is not None:
            delay = label_times - timestamps
            self.delay_neg_count += int((delay < 0).sum())
            # Clip to >=0 for histogram
            delay_nn = np.clip(delay, 0, None)
            buckets = np.searchsorted(DELAY_HIST_BINS, delay_nn)
            np.add.at(self.delay_hist_all, buckets, 1)
            pos_mask = (label_types == 2)
            if pos_mask.any():
                np.add.at(self.delay_hist_pos, buckets[pos_mask], 1)
                self.delay_sum += float(delay_nn[pos_mask].sum())
                self.delay_count += int(pos_mask.sum())

    def finalize(self) -> Dict[str, Any]:
        r: Dict[str, Any] = {
            'n_rows': self.n,
            'timestamp_min': self.ts_min,
            'timestamp_max': self.ts_max,
            'timestamp_span_seconds': (self.ts_max - self.ts_min) if (
                self.ts_max is not None and self.ts_min is not None) else None,
            'timestamp_span_days': (self.ts_max - self.ts_min) / 86400.0 if (
                self.ts_max is not None and self.ts_min is not None) else None,
            'hour_histogram': self.hour_counts.tolist(),
            'dow_histogram': self.dow_counts.tolist(),
        }
        if self.label_counts:
            r['label_type_counts'] = dict(sorted(self.label_counts.items()))
            total = sum(self.label_counts.values())
            r['positive_rate_label2'] = (
                self.label_counts.get(2, 0) / max(1, total))
            r['hour_positive_counts'] = self.hour_x_pos.tolist()
            # Per-hour positive rate
            with np.errstate(divide='ignore', invalid='ignore'):
                hourly_pos_rate = self.hour_x_pos / np.maximum(self.hour_counts, 1)
            r['hour_positive_rate'] = hourly_pos_rate.tolist()
        if self.delay_count > 0 or self.delay_hist_all.sum() > 0:
            r['delay_bins'] = (['<=' + str(b) + 's' for b in DELAY_HIST_BINS]
                               + ['>' + str(DELAY_HIST_BINS[-1]) + 's'])
            r['delay_hist_all'] = self.delay_hist_all.tolist()
            r['delay_hist_pos'] = self.delay_hist_pos.tolist()
            r['delay_negative_count'] = self.delay_neg_count
            if self.delay_count > 0:
                r['delay_pos_mean_seconds'] = self.delay_sum / self.delay_count
        return r


# ────────────────────────── Column-type detection ────────────────────────────

def detect_column_type(arrow_type) -> Tuple[str, str]:
    """Return ('scalar'|'array'|'unknown', dtype_kind) for an Arrow type."""
    if pa.types.is_list(arrow_type):
        inner = arrow_type.value_type
        if pa.types.is_integer(inner):
            return 'array', 'i'
        if pa.types.is_floating(inner):
            return 'array', 'f'
        return 'array', '?'
    if pa.types.is_integer(arrow_type):
        return 'scalar', 'i'
    if pa.types.is_floating(arrow_type):
        return 'scalar', 'f'
    return 'unknown', '?'


# ────────────────────────── Main streaming loop ──────────────────────────────

def stream_eda(
    data_dir: str,
    output_path: str,
    log_dir: Optional[str],
    is_training: bool,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    """Walk every parquet file, accumulate stats, dump to output_path."""
    setup_logger(log_dir)
    logging.info('=== EDA START ===')
    logging.info(f'data_dir={data_dir} is_training={is_training}')

    import glob
    files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No .parquet under {data_dir}')
    if max_files is not None:
        files = files[:max_files]
    logging.info(f'Found {len(files)} parquet files; first={files[0]}')

    # Initialize accumulators on the first batch (need column names + types).
    accumulators: Dict[str, Any] = {}
    sample_accum = SampleAccum()
    col_meta: Dict[str, Tuple[str, str]] = {}
    col_skip_warned: set = set()

    t0 = time.time()
    rows_seen = 0
    for fi, fpath in enumerate(files):
        try:
            pf = pq.ParquetFile(fpath)
        except Exception as exc:
            logging.warning(f'Failed to open {fpath}: {exc}; skipping')
            continue
        if fi == 0:
            # Initialize accumulators by scanning the schema
            schema = pf.schema_arrow
            for name, atype in zip(schema.names, schema.types):
                kind, dk = detect_column_type(atype)
                col_meta[name] = (kind, dk)
                if kind == 'scalar':
                    accumulators[name] = ScalarAccum(name, dk)
                elif kind == 'array':
                    accumulators[name] = ArrayAccum(name, dk)
            logging.info(
                f'Schema has {len(col_meta)} columns: '
                f'{sum(1 for v in col_meta.values() if v[0] == "scalar")} scalar, '
                f'{sum(1 for v in col_meta.values() if v[0] == "array")} array, '
                f'{sum(1 for v in col_meta.values() if v[0] == "unknown")} unknown')

        for rg_idx in range(pf.metadata.num_row_groups):
            try:
                batch = pf.read_row_group(rg_idx)
            except Exception as exc:
                logging.warning(
                    f'Failed to read RG {rg_idx} from {fpath}: {exc}; skipping')
                continue
            B = batch.num_rows
            rows_seen += B

            # Pull labels and timestamps once. Tables hand back ChunkedArrays
            # so flatten via combine_chunks (no-op when there's just one chunk).
            def _col_int64(name: str):
                if name not in batch.schema.names:
                    return None
                c = batch.column(name)
                if hasattr(c, 'combine_chunks'):
                    c = c.combine_chunks()
                return c.to_numpy(zero_copy_only=False).astype(np.int64)

            timestamps = _col_int64('timestamp')
            label_types = _col_int64('label_type') if is_training else None
            label_times = _col_int64('label_time') if is_training else None

            # Binary label for 1-D AUC: 1 if label_type==2.
            binary_label = None
            if label_types is not None:
                binary_label = (label_types == 2).astype(np.int64)

            # Sample-level accumulator
            if timestamps is not None:
                sample_accum.update(timestamps, label_types, label_times)

            # Per-column accumulator updates
            for cname, (kind, dk) in col_meta.items():
                if kind == 'unknown':
                    if cname not in col_skip_warned:
                        col_skip_warned.add(cname)
                        logging.warning(f'Skipping unknown-typed column {cname}')
                    continue
                col = batch.column(cname)
                # ``read_row_group`` returns a Table -> column is ChunkedArray.
                # combine_chunks() collapses into a single Array so we can
                # access .offsets / .values directly without per-chunk merges.
                if hasattr(col, 'combine_chunks'):
                    col = col.combine_chunks()
                if kind == 'scalar':
                    arr = col.fill_null(0 if dk in 'iu' else float('nan')).to_numpy(
                        zero_copy_only=False)
                    if dk in 'iu':
                        arr = arr.astype(np.int64)
                    else:
                        arr = arr.astype(np.float64)
                    accumulators[cname].update(arr, binary_label)
                else:
                    # list<int|float>: pull offsets + flat values
                    offsets = col.offsets.to_numpy().astype(np.int64)
                    flat = col.values.to_numpy(zero_copy_only=False)
                    if dk in 'iu':
                        flat = flat.astype(np.int64)
                    else:
                        flat = flat.astype(np.float64)
                    accumulators[cname].update(offsets, flat, B, binary_label)

        if (fi + 1) % 50 == 0 or fi == len(files) - 1:
            elapsed = time.time() - t0
            rate = rows_seen / max(1e-6, elapsed)
            logging.info(
                f'Processed {fi + 1}/{len(files)} files, '
                f'{rows_seen:,} rows ({rate:.0f} rows/s, '
                f'{elapsed:.0f}s elapsed)')

    # Finalize all accumulators.
    feature_report: Dict[str, Any] = {}
    for cname, acc in accumulators.items():
        feature_report[cname] = acc.finalize()
        feature_report[cname]['column_kind'] = col_meta[cname][0]
        feature_report[cname]['dtype_kind'] = col_meta[cname][1]

    full_report = {
        'meta': {
            'data_dir': data_dir,
            'n_files': len(files),
            'n_rows_seen': rows_seen,
            'elapsed_seconds': time.time() - t0,
            'is_training': is_training,
            'unique_cap': UNIQUE_CAP,
            'auc_sample_cap': AUC_SAMPLE_CAP,
            'len_sample_cap': LEN_SAMPLE_CAP,
        },
        'sample_stats': sample_accum.finalize(),
        'features': feature_report,
    }

    # Dump JSON.
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(full_report, f, indent=2, default=_json_default)
    logging.info(f'Wrote JSON report to {output_path}')

    # Print summary.
    _log_summary(full_report)
    logging.info('=== EDA END ===')

    return full_report


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _log_summary(report: Dict[str, Any]) -> None:
    """Print human-readable highlights at INFO level."""
    s = report['sample_stats']
    logging.info('==== SAMPLE-LEVEL SUMMARY ====')
    logging.info(f"Total rows: {s['n_rows']:,}")
    if s.get('timestamp_min') is not None:
        logging.info(
            f"Timestamp range: {s['timestamp_min']} - {s['timestamp_max']} "
            f"({s.get('timestamp_span_days', 0):.2f} days)")
    if 'label_type_counts' in s:
        logging.info(f"Label type counts: {s['label_type_counts']}")
        logging.info(f"Positive rate (label_type==2): "
                     f"{s['positive_rate_label2']:.4f}")
    if 'hour_histogram' in s:
        logging.info('Hour-of-day histogram:')
        for h, c in enumerate(s['hour_histogram']):
            bar = '#' * int(c * 60 / max(s['hour_histogram']))
            logging.info(f"  hour={h:2d} count={c:>10,} {bar}")
    if 'hour_positive_rate' in s:
        logging.info('Hour-of-day positive rate (label_type==2):')
        for h, r in enumerate(s['hour_positive_rate']):
            logging.info(f"  hour={h:2d} pos_rate={r:.4f}")
    if 'dow_histogram' in s:
        logging.info(f"Day-of-week histogram (Mon=0..Sun=6): "
                     f"{s['dow_histogram']}")
    if 'delay_hist_all' in s:
        logging.info('Conversion delay distribution (label_time - timestamp):')
        logging.info(f"  bins: {s['delay_bins']}")
        logging.info(f"  all : {s['delay_hist_all']}")
        logging.info(f"  pos : {s['delay_hist_pos']}")
        logging.info(f"  negative-delay rows: {s['delay_negative_count']}")
        if 'delay_pos_mean_seconds' in s:
            logging.info(
                f"  positive-sample mean delay: "
                f"{s['delay_pos_mean_seconds']:.1f}s")

    feats = report['features']
    logging.info('')
    logging.info('==== PER-FEATURE SUMMARY ====')

    # Group by kind + sort interesting subsets
    scalars = {k: v for k, v in feats.items() if v.get('column_kind') == 'scalar'}
    arrays = {k: v for k, v in feats.items() if v.get('column_kind') == 'array'}

    # High null/zero scalars
    high_null = sorted(
        [(k, v) for k, v in scalars.items() if v.get('null_rate', 0) > 0.5
         or v.get('zero_rate', 0) > 0.95],
        key=lambda kv: -kv[1].get('zero_rate', 0))[:20]
    if high_null:
        logging.info('High null/zero-rate scalar features (top 20):')
        for k, v in high_null:
            logging.info(
                f"  {k}: zero_rate={v.get('zero_rate', 0):.3f} "
                f"null_rate={v.get('null_rate', 0):.3f} "
                f"unique~{v.get('unique_sampled', 0)}")

    # 1-D AUC ranking (training only)
    auc_ranked = sorted(
        [(k, v) for k, v in feats.items()
         if 'auc_signal' in v or 'auc_1d' in v],
        key=lambda kv: -kv[1].get('auc_signal',
                                  abs(kv[1].get('auc_1d', 0.5) - 0.5) * 2))
    if auc_ranked:
        logging.info('Top 25 features by 1-D AUC signal '
                     '(|AUC - 0.5| * 2, higher = stronger single-feature signal):')
        for k, v in auc_ranked[:25]:
            auc = v.get('auc_1d') or v.get('auc_best') or 0.5
            sig = v.get('auc_signal', abs(auc - 0.5) * 2)
            logging.info(
                f"  {k}: auc={auc:.4f} signal={sig:.4f} "
                f"kind={v.get('column_kind')}")

    # Array length summary
    if arrays:
        logging.info('Array column length summary:')
        for k, v in sorted(arrays.items()):
            logging.info(
                f"  {k}: len_mean={v.get('len_mean', 0):.1f} "
                f"len_p95={v.get('len_p95', 0):.0f} "
                f"len_max={v.get('len_max', 0)} "
                f"empty_rate={v.get('empty_rate', 0):.3f} "
                f"value_max={v.get('value_max', 0)}")

    # Full feature table -- one TSV line per feature. Designed so the
    # entire EDA result is recoverable by reading the log alone (no need
    # to download the JSON). Lines are prefixed with "TSV" so a single
    # ``grep '^TSV ' eda.log`` extracts a clean table.
    _log_feature_table_tsv(feats)
    # Likewise dump the sample-level histograms / delay distribution as
    # KV lines that are easy to copy/paste back.
    _log_sample_table_tsv(s)


def _fmt(v: Any) -> str:
    """TSV-friendly formatter (numeric -> short repr, None -> '')."""
    if v is None:
        return ''
    if isinstance(v, float):
        if abs(v) >= 1e6 or (abs(v) < 1e-3 and v != 0):
            return f'{v:.6g}'
        return f'{v:.6f}'.rstrip('0').rstrip('.')
    if isinstance(v, (list, tuple)):
        return '|'.join(_fmt(x) for x in v)
    return str(v)


def _log_feature_table_tsv(features: Dict[str, Any]) -> None:
    """Emit one TSV line per feature. Columns are constant across rows so a
    spreadsheet paste lines up; missing values appear as empty fields.
    """
    logging.info('')
    logging.info('==== FULL FEATURE TABLE (lines prefixed with "TSV ") ====')
    logging.info(
        'TSV  name\tkind\tdtype\tn_total\tnull_rate\tzero_or_empty_rate\t'
        'min\tmax\tmean\tstd\tunique_sampled\tunique_capped\t'
        'len_mean\tlen_p95\tlen_max\tflat_zero_rate\t'
        'auc_1d_or_best\tauc_signal\tauc_via_mean\tauc_via_first_nz\tauc_via_max')
    # Sort by AUC signal (descending) so the most predictive features
    # come first; ties break by name.
    def keyfn(item):
        v = item[1]
        sig = v.get('auc_signal')
        if sig is None and 'auc_1d' in v:
            sig = abs(v['auc_1d'] - 0.5) * 2
        return (-(sig or 0.0), item[0])
    for name, v in sorted(features.items(), key=keyfn):
        kind = v.get('column_kind', '?')
        dtype = v.get('dtype_kind', '?')
        n_total = v.get('n_total', 0)
        null_rate = v.get('null_rate', 0)
        # scalars expose zero_rate, arrays expose empty_rate -- merge into one column
        zoer = v.get('zero_rate', v.get('empty_rate', None))
        mn = v.get('min', v.get('value_min', None))
        mx = v.get('max', v.get('value_max', None))
        mean = v.get('mean', None)
        std = v.get('std', None)
        uniq = v.get('unique_sampled', None)
        uniq_capped = v.get('unique_capped', None)
        len_mean = v.get('len_mean', None)
        len_p95 = v.get('len_p95', None)
        len_max = v.get('len_max', None)
        flat_zero = v.get('flat_zero_rate', None)
        auc_main = v.get('auc_1d', v.get('auc_best'))
        sig = v.get('auc_signal')
        auc_mean = v.get('auc_via_mean')
        auc_first = v.get('auc_via_first_nz')
        auc_max = v.get('auc_via_max')
        logging.info(
            f'TSV  {name}\t{kind}\t{dtype}\t{n_total}\t{_fmt(null_rate)}\t'
            f'{_fmt(zoer)}\t{_fmt(mn)}\t{_fmt(mx)}\t{_fmt(mean)}\t{_fmt(std)}\t'
            f'{_fmt(uniq)}\t{_fmt(uniq_capped)}\t'
            f'{_fmt(len_mean)}\t{_fmt(len_p95)}\t{_fmt(len_max)}\t'
            f'{_fmt(flat_zero)}\t{_fmt(auc_main)}\t{_fmt(sig)}\t'
            f'{_fmt(auc_mean)}\t{_fmt(auc_first)}\t{_fmt(auc_max)}')


def _log_sample_table_tsv(sample: Dict[str, Any]) -> None:
    """Emit sample-level histograms as KV lines (each ``KV key=value`` is
    easy to extract via ``grep '^KV '``).
    """
    logging.info('')
    logging.info('==== SAMPLE-LEVEL KV (lines prefixed with "KV ") ====')
    flat_keys = ['n_rows', 'timestamp_min', 'timestamp_max',
                 'timestamp_span_seconds', 'timestamp_span_days',
                 'positive_rate_label2', 'delay_negative_count',
                 'delay_pos_mean_seconds']
    for k in flat_keys:
        if k in sample:
            logging.info(f'KV  {k}\t{_fmt(sample[k])}')
    if 'label_type_counts' in sample:
        for lt, c in sample['label_type_counts'].items():
            logging.info(f'KV  label_type_{lt}_count\t{c}')
    if 'hour_histogram' in sample:
        for h, c in enumerate(sample['hour_histogram']):
            logging.info(f'KV  hour_count_{h:02d}\t{c}')
    if 'hour_positive_rate' in sample:
        for h, r in enumerate(sample['hour_positive_rate']):
            logging.info(f'KV  hour_pos_rate_{h:02d}\t{_fmt(r)}')
    if 'dow_histogram' in sample:
        for d, c in enumerate(sample['dow_histogram']):
            logging.info(f'KV  dow_count_{d}\t{c}')
    if 'delay_hist_all' in sample and 'delay_bins' in sample:
        bins = sample['delay_bins']
        for b, c_all, c_pos in zip(bins, sample['delay_hist_all'],
                                    sample.get('delay_hist_pos', [0] * len(bins))):
            logging.info(f'KV  delay_{b}_all\t{c_all}')
            logging.info(f'KV  delay_{b}_pos\t{c_pos}')


# ─────────────────────────────── Entry point ────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description='TAAC2026 PCVR EDA')
    ap.add_argument('--data_dir', type=str, default=None,
                    help='Parquet directory (env TRAIN_DATA_PATH overrides)')
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Where to dump eda_report.json '
                         '(env TRAIN_CKPT_PATH overrides)')
    ap.add_argument('--log_dir', type=str, default=None,
                    help='Where to write eda.log (env TRAIN_LOG_PATH overrides)')
    ap.add_argument('--mode', type=str, default='train',
                    choices=['train', 'eval'],
                    help='train: derive labels from label_type==2; eval: '
                         'skip label-dependent stats')
    ap.add_argument('--max_files', type=int, default=None,
                    help='Limit number of files (testing / smoke runs)')
    args = ap.parse_args()

    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.output_dir = os.environ.get('TRAIN_CKPT_PATH', args.output_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)

    if not args.data_dir:
        raise ValueError('data_dir not set (CLI or TRAIN_DATA_PATH env)')
    if not args.output_dir:
        raise ValueError('output_dir not set (CLI or TRAIN_CKPT_PATH env)')

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        'eda_report_train.json' if args.mode == 'train' else 'eda_report_eval.json')

    stream_eda(
        data_dir=args.data_dir,
        output_path=output_path,
        log_dir=args.log_dir,
        is_training=(args.mode == 'train'),
        max_files=args.max_files,
    )


if __name__ == '__main__':
    main()
