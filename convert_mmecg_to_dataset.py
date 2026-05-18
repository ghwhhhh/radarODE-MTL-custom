import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
import torch
import torch.nn.functional as F


def _resolve_dir(input_path: str, must_exist: bool = True) -> Path:
    """Resolve directory from multiple common bases so the script is robust to cwd changes."""
    raw = Path(input_path)
    script_dir = Path(__file__).resolve().parent
    candidates = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([
            Path.cwd() / raw,
            script_dir / raw,
            script_dir.parent / raw,
        ])

    # Remove duplicates while preserving order.
    uniq = []
    seen = set()
    for c in candidates:
        r = c.resolve()
        if str(r) not in seen:
            seen.add(str(r))
            uniq.append(r)

    if must_exist:
        for c in uniq:
            if c.exists() and c.is_dir():
                return c
        tried = "\n".join(f"  - {p}" for p in uniq)
        raise FileNotFoundError(
            f"Directory not found for '{input_path}'. Tried:\n{tried}"
        )

    return uniq[0]


def _as_scalar(x):
    arr = np.asarray(x)
    if arr.size == 0:
        return ""
    v = arr.reshape(-1)[0]
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    return str(v)


def _safe_norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def _cwt_sst_segment(rcg_seg_800x50: np.ndarray, target_freq=71, target_time=120, device="cuda") -> np.ndarray:
    """Build SST-like tensor with shape (50, 71, 120) from one 4s radar segment using GPU CWT."""
    sst = np.zeros((50, target_freq, target_time), dtype=np.float32)
    sig_len = rcg_seg_800x50.shape[0]
    
    for ch in range(50):
        sig = torch.from_numpy(rcg_seg_800x50[:, ch].astype(np.float32)).to(device)
        
        # GPU-based Morlet CWT via Fourier domain
        sig_fft = torch.fft.rfft(sig)
        coef = torch.zeros((target_freq, sig_len), dtype=torch.complex64, device=device)
        
        for scale_idx, scale in enumerate(np.arange(1, target_freq + 1)):
            # Morlet wavelet in frequency domain (simplified)
            freq = torch.fft.rfftfreq(sig_len, d=1.0).to(device)
            # Morlet: exp(-(freq - fc)^2 / (2*sigma^2)) for scale
            fc = 0.5 / scale  # center frequency
            sigma = 0.5 / scale  # bandwidth
            psi = torch.exp(-((freq - fc) ** 2) / (2 * sigma ** 2))
            psi = psi / torch.sqrt(torch.sum(psi ** 2) + 1e-12)
            coef[scale_idx] = torch.fft.irfft(sig_fft * psi, n=sig_len)
        
        # Magnitude and normalize
        mag = torch.abs(coef).cpu().numpy()
        
        # Resample in time using GPU interpolation
        # F.interpolate requires (N, C, H, W) format for bilinear mode
        mag_t = torch.from_numpy(mag)[None, None, :, :].to(device)  # (1, 1, 71, 800)
        mag_resized = F.interpolate(
            mag_t, 
            size=(target_freq, target_time), 
            mode='bilinear', 
            align_corners=False
        )
        mag = mag_resized.squeeze(0).squeeze(0).cpu().numpy()
        
        sst[ch] = _safe_norm01(mag)
    
    return sst


def _detect_peaks(ecg_800: np.ndarray, fs: int) -> np.ndarray:
    ecg_800 = np.asarray(ecg_800, dtype=np.float32).reshape(-1)
    # Robust default: normalise then detect local peaks with physiologic min distance.
    z = ecg_800 - np.median(ecg_800)
    mad = np.median(np.abs(z)) + 1e-6
    z = z / mad
    peaks, _ = find_peaks(z, distance=int(0.35 * fs), prominence=0.6)
    return peaks.astype(np.int32)


def _pick_one_cycle(ecg_800: np.ndarray, peaks: np.ndarray, max_len=260, fallback_len=200) -> np.ndarray:
    if len(peaks) >= 2:
        center = len(ecg_800) / 2.0
        pair_idx = np.argmin(np.abs(((peaks[:-1] + peaks[1:]) / 2.0) - center))
        s, e = int(peaks[pair_idx]), int(peaks[pair_idx + 1])
        cyc = ecg_800[s:e]
    else:
        # Fallback if peak detection is weak: take centered fixed-length segment.
        half = fallback_len // 2
        c = len(ecg_800) // 2
        cyc = ecg_800[max(0, c - half): min(len(ecg_800), c + half)]

    cyc = np.asarray(cyc, dtype=np.float32).reshape(-1)
    if cyc.size == 0:
        cyc = np.zeros((fallback_len,), dtype=np.float32)
    if cyc.size > max_len:
        cyc = resample(cyc, max_len)
    return cyc.astype(np.float32)


def convert_one_mat(mat_path: Path, out_root: Path, seg_sec: float, step_sec: float, fs: int, device: str) -> int:
    obj = loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    data = obj["data"][0, 0]

    rcg = np.asarray(data.RCG, dtype=np.float32)  # (N, 50)
    ecg = np.asarray(data.ECG, dtype=np.float32).reshape(-1)  # (N,)

    sid = _as_scalar(data.id)
    status = _as_scalar(data.physistatus).strip()
    file_idx = mat_path.stem

    out_dir = out_root / f"obj{sid}_{status}_{file_idx}_"
    out_dir.mkdir(parents=True, exist_ok=True)

    win = int(seg_sec * fs)
    step = int(step_sec * fs)
    if rcg.shape[0] < win:
        return 0

    seg_count = 0
    for start in range(0, rcg.shape[0] - win + 1, step):
        end = start + win
        rcg_seg = rcg[start:end, :]  # (800, 50) for 4s@200Hz
        ecg_seg = ecg[start:end]      # (800,)

        sst = _cwt_sst_segment(rcg_seg, target_freq=71, target_time=120, device=device)
        peaks = _detect_peaks(ecg_seg, fs=fs)

        anchor = np.zeros((win,), dtype=np.float32)
        anchor[np.clip(peaks, 0, win - 1)] = 1.0

        cycle = _pick_one_cycle(ecg_seg, peaks, max_len=260, fallback_len=200)

        np.save(out_dir / f"sst_seg_{seg_count}.npy", sst)
        np.save(out_dir / f"anchor_seg_{seg_count}.npy", anchor)
        np.save(out_dir / f"ecg_seg_{seg_count}.npy", cycle)
        seg_count += 1

    return seg_count


def main():
    parser = argparse.ArgumentParser(description="Convert MMECG .mat files to radarODE/radarODE-MTL Dataset format (GPU-accelerated).")
    parser.add_argument("--mat_root", type=str, required=True, help="Folder containing 1.mat..91.mat")
    parser.add_argument("--out_root", type=str, default="Dataset", help="Output Dataset root")
    parser.add_argument("--fs", type=int, default=200, help="Sampling rate of ECG/RCG in source MAT")
    parser.add_argument("--seg_sec", type=float, default=4.0, help="Segment duration in seconds")
    parser.add_argument("--step_sec", type=float, default=4.0, help="Sliding step in seconds")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N mats (0 means all)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID (0 for cuda:0, -1 for CPU)")
    args = parser.parse_args()

    # Device setup
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}"
        print(f"Using GPU: {device} ({torch.cuda.get_device_name(args.gpu_id)})")
    else:
        device = "cpu"
        print("GPU not available or disabled, using CPU")

    mat_root = _resolve_dir(args.mat_root, must_exist=True)
    out_root = _resolve_dir(args.out_root, must_exist=False)
    out_root.mkdir(parents=True, exist_ok=True)

    mats = sorted(mat_root.glob("*.mat"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if args.limit > 0:
        mats = mats[: args.limit]

    if not mats:
        raise FileNotFoundError(
            f"No .mat files found in {mat_root}. "
            "Please check --mat_root points to a folder containing *.mat files."
        )

    total_segs = 0
    for i, mat_file in enumerate(mats, start=1):
        n = convert_one_mat(
            mat_path=mat_file,
            out_root=out_root,
            seg_sec=args.seg_sec,
            step_sec=args.step_sec,
            fs=args.fs,
            device=device,
        )
        total_segs += n
        print(f"[{i:03d}/{len(mats):03d}] {mat_file.name}: {n} segments")

    print("=" * 64)
    print(f"Done. Output: {out_root}")
    print(f"Processed mats: {len(mats)} | Total segments: {total_segs}")
    print("Expected per sample shapes:")
    print("  sst_seg_k.npy    -> (50, 71, 120)")
    print("  anchor_seg_k.npy -> (800,)")
    print("  ecg_seg_k.npy    -> (L,), L<=260")


if __name__ == "__main__":
    main()
