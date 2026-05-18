"""
preprocess_demo.py
------------------
Convert db_records sessions into (sst_seg, ecg_seg, anchor_seg) .npy triplets
compatible with SpectrumECGDataset / radarODE-MTL training.

Data format contract (matching the original MMECG dataset convention):
  sst_seg_N.npy   : float32  (50, 71, 120)  -- CWT time-frequency map
  ecg_seg_N.npy   : float32  (L,)  L <= 255 -- ONE single cardiac cycle at
                     native ECG fs (variable length; the dataset loader pads to
                     260 with -10 so that L encodes the PPI in ECG samples)
  anchor_seg_N.npy: float32  (800,)          -- R-peak indicator at 200 Hz basis
                     (1.0 at peak positions, 0 elsewhere)
"""

import argparse
import os
import pickle
import zlib

import numpy as np
import pandas as pd
import pywt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks


def _ts_to_sec(series):
    dt = pd.to_datetime(series, errors='coerce')
    if dt.isna().any():
        raise ValueError('Failed to parse some timestamps')
    raw = dt.astype('int64').to_numpy()
    if raw[0] < 1e17:
        return raw / 1e6
    else:
        return raw / 1e9


def _infer_fs(time_sec):
    dt = np.diff(time_sec)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError('Cannot infer sampling rate')
    return 1.0 / float(np.median(dt))


def _interp_resample_1d(x, src_t, dst_t):
    f = interp1d(src_t, x, kind='linear', bounds_error=False, fill_value='extrapolate')
    return f(dst_t)


def _minmax_01(x, eps=1e-8):
    lo, hi = np.min(x), np.max(x)
    if float(hi - lo) < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _make_cwt_sst(signal_1d, scales):
    coef, _ = pywt.cwt(signal_1d, scales, 'morl')
    return np.abs(coef)


def _detect_rpeaks(ecg, fs):
    lo = 0.5 / (fs / 2)
    hi = min(40.0 / (fs / 2), 0.99)
    b, a = butter(2, [lo, hi], btype='band')
    ecg_f = filtfilt(b, a, ecg)
    min_dist = int(0.35 * fs)
    height_thr = np.percentile(ecg_f, 70)
    peaks, _ = find_peaks(ecg_f, distance=min_dist, height=height_thr)
    return peaks


def process_single_record(
    record_dir,
    out_dir,
    win_sec=4,
    desired_radar_fs=30,
    max_segments=None,
    range_bin_start=7,
    seg_offset=0,
):
    print(f"Processing: {record_dir}")
    os.makedirs(out_dir, exist_ok=True)

    radar_fft_path = os.path.join(record_dir, 'radar_rFFTs.zlib')
    radar_ts_path  = os.path.join(record_dir, 'radar_timestamps.csv')
    ecg_path       = os.path.join(record_dir, 'movesense_ecg.csv')

    for p in (radar_fft_path, radar_ts_path, ecg_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    radar_ts_df = pd.read_csv(radar_ts_path, header=None)
    radar_time_sec = _ts_to_sec(radar_ts_df.iloc[:, 0])

    blob = zlib.decompress(open(radar_fft_path, 'rb').read())
    obj  = pickle.loads(blob)
    if not isinstance(obj, list) or len(obj) < 1:
        raise ValueError('Unexpected radar_rFFTs content')
    radar_arr = obj[0]
    if not isinstance(radar_arr, np.ndarray) or radar_arr.ndim != 3:
        raise ValueError(f'Unexpected radar array shape: {getattr(radar_arr, "shape", None)}')

    T = min(len(radar_time_sec), radar_arr.shape[0])
    radar_time_sec = radar_time_sec[:T]
    radar_arr      = radar_arr[:T]

    radar_mag = np.abs(radar_arr).mean(axis=1)
    if radar_mag.shape[1] < 50:
        raise ValueError(f'Need >= 50 range bins, got {radar_mag.shape[1]}')
    rb_s = int(range_bin_start)
    rb_e = rb_s + 50
    if rb_e > radar_mag.shape[1]:
        rb_s = max(0, radar_mag.shape[1] - 50)
        rb_e = rb_s + 50
    radar_50 = radar_mag[:, rb_s:rb_e]

    fs_radar = _infer_fs(radar_time_sec)
    win_len  = int(round(win_sec * fs_radar))
    if win_len <= 4:
        raise ValueError(f'Window too small: win_len={win_len}, fs_radar={fs_radar}')

    ecg_df  = pd.read_csv(ecg_path)
    ecg_time_sec = _ts_to_sec(ecg_df.iloc[:, 0])
    ecg_val = ecg_df.iloc[:, 1].astype(float).to_numpy()
    fs_ecg  = _infer_fs(ecg_time_sec)

    scales = np.arange(1, 72, dtype=float)

    n_segs = T // win_len
    if max_segments is not None:
        n_segs = min(n_segs, int(max_segments))

    print(f"  fs_radar~{fs_radar:.1f} Hz  fs_ecg~{fs_ecg:.1f} Hz  win_len={win_len}  total_segs={n_segs}")

    written = 0
    for seg_idx in range(n_segs):
        s  = seg_idx * win_len
        e  = s + win_len
        radar_win_t = radar_time_sec[s:e].astype(float)
        t0 = radar_win_t[0]
        t1 = radar_win_t[-1]
        if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
            continue

        # Use real radar timestamps for resampling to avoid timing drift from uniform-grid assumptions.
        uniq_t, uniq_idx = np.unique(radar_win_t, return_index=True)
        if uniq_t.size < 4:
            continue

        dst_len = int(desired_radar_fs * win_sec)
        dst_t   = np.linspace(t0, t1, num=dst_len)

        sst_ch = []
        for ch in range(50):
            sig = radar_50[s:e, ch].astype(float)
            sig_rs = _interp_resample_1d(sig[uniq_idx], uniq_t, dst_t)
            tf     = _make_cwt_sst(sig_rs, scales)
            sst_ch.append(_minmax_01(tf))
        sst = np.stack(sst_ch, axis=0).astype(np.float32)

        mask      = (ecg_time_sec >= t0) & (ecg_time_sec <= t1)
        ecg_win_t = ecg_time_sec[mask].astype(float)
        ecg_win   = ecg_val[mask]

        if len(ecg_win) < 10:
            continue

        peaks_win = _detect_rpeaks(ecg_win, fs_ecg)

        anchor = np.zeros(800, dtype=np.float32)
        win_samples_200hz = int(win_sec * 200)
        for pk in peaks_win:
            pk_t = float(ecg_win_t[int(pk)])
            phase = (pk_t - t0) / (t1 - t0)
            pk_200 = int(round(phase * (win_samples_200hz - 1)))
            pk_200 = max(0, pk_200)
            pk_200 = min(pk_200, 799)
            anchor[pk_200] = 1.0

        MAX_CYCLE_LEN = 255
        if len(peaks_win) >= 2:
            cycle_s = int(peaks_win[0])
            cycle_e = int(peaks_win[1])
            cycle_len = cycle_e - cycle_s
            if 30 <= cycle_len <= MAX_CYCLE_LEN:
                ecg_seg = ecg_win[cycle_s:cycle_e].astype(np.float32)
            else:
                ecg_seg = ecg_win[cycle_s: cycle_s + MAX_CYCLE_LEN].astype(np.float32)
        else:
            ecg_seg = ecg_win[:MAX_CYCLE_LEN].astype(np.float32)

        out_idx = seg_offset + written
        np.save(os.path.join(out_dir, f"sst_seg_{out_idx}.npy"),    sst)
        np.save(os.path.join(out_dir, f"ecg_seg_{out_idx}.npy"),    ecg_seg)
        np.save(os.path.join(out_dir, f"anchor_seg_{out_idx}.npy"), anchor)
        written += 1

    print(f"  -> wrote {written} segments to {out_dir}")
    return written


def process_all(
    db_root,
    out_root,
    win_sec=4,
    desired_radar_fs=30,
    max_segments_per_session=None,
    range_bin_start=7,
    include_postures=None,
):
    include_postures_lc = None
    if include_postures:
        include_postures_lc = {p.strip().lower() for p in include_postures if p and p.strip()}

    sessions = []
    for pdir in sorted(os.listdir(db_root)):
        ppath = os.path.join(db_root, pdir)
        if not os.path.isdir(ppath) or not pdir.startswith('P'):
            continue
        for posture in sorted(os.listdir(ppath)):
            pospath = os.path.join(ppath, posture)
            if not os.path.isdir(pospath):
                continue
            if include_postures_lc is not None and posture.lower() not in include_postures_lc:
                continue
            for activity in sorted(os.listdir(pospath)):
                actpath = os.path.join(pospath, activity)
                req = ['radar_rFFTs.zlib', 'radar_timestamps.csv', 'movesense_ecg.csv']
                if all(os.path.exists(os.path.join(actpath, r)) for r in req):
                    sessions.append((pdir, posture, activity, actpath))

    print(f"Found {len(sessions)} sessions.")
    for dataset_id, (pdir, posture, activity, actpath) in enumerate(sessions, start=1):
        tag = f"{posture[:2]}_{activity[:2]}"
        out_dir = os.path.join(out_root, f"{pdir}_{tag}_{dataset_id}_")
        try:
            process_single_record(
                record_dir=actpath,
                out_dir=out_dir,
                win_sec=win_sec,
                desired_radar_fs=desired_radar_fs,
                max_segments=max_segments_per_session,
                range_bin_start=range_bin_start,
            )
        except Exception as exc:
            print(f"  [SKIP] {actpath}: {exc}")

    print("Batch preprocessing complete.")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',         choices=['single', 'batch'], default='single')
    p.add_argument('--record_dir',   default='db_records/P001/Sitting/Rest')
    p.add_argument('--out_dir',      default='Dataset/obj1_DB_1_')
    p.add_argument('--db_root',      default='db_records')
    p.add_argument('--out_root',     default='Dataset')
    p.add_argument('--win_sec',          type=int,  default=4)
    p.add_argument('--desired_radar_fs', type=int,  default=30)
    p.add_argument('--max_segments',     type=int,  default=None)
    p.add_argument('--range_bin_start',  type=int,  default=7)
    p.add_argument(
        '--include_postures',
        type=str,
        default='',
        help='Batch mode only. Comma-separated posture names to keep, e.g. "Sitting" or "Sitting,Standing". Empty means all postures.',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    include_postures = [x.strip() for x in args.include_postures.split(',') if x.strip()]
    if args.mode == 'batch':
        process_all(
            db_root=args.db_root,
            out_root=args.out_root,
            win_sec=args.win_sec,
            desired_radar_fs=args.desired_radar_fs,
            max_segments_per_session=args.max_segments,
            range_bin_start=args.range_bin_start,
            include_postures=include_postures,
        )
    else:
        process_single_record(
            record_dir=args.record_dir,
            out_dir=args.out_dir,
            win_sec=args.win_sec,
            desired_radar_fs=args.desired_radar_fs,
            max_segments=args.max_segments,
            range_bin_start=args.range_bin_start,
        )