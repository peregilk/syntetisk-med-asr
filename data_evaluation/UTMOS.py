#!/usr/bin/env python3
"""
Run UTMOSv2 on all audio files in a folder (optionally recursive), and write:
  - results.csv
  - summary.md (with tables)
  - report.html
"""

from __future__ import annotations

import argparse
import logging
import math
import multiprocessing as mp
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
import torch
import utmosv2
from tabulate import tabulate
from tqdm import tqdm


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".caf"}
TARGET_SR = 16000


def mos_to_0_100(mos: float) -> float:
    """Convert MOS score (1-5) to 0-100 scale."""
    return (mos - 1) * 25


@dataclass
class FileResult:
    rel_path: str
    ok: bool
    mos: Optional[float]
    mos_0_100: Optional[float]
    duration_s: Optional[float]
    input_sr: Optional[int]
    channels: Optional[int]
    error: Optional[str]


def find_audio_files(root: Path, recursive: bool) -> List[Path]:
    it = root.rglob("*") if recursive else root.glob("*")
    files = [p for p in it if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return sorted(files)


def load_audio(path: Path) -> Tuple[np.ndarray, int, int, float]:
    """
    Load audio into mono float32 numpy array.
    Uses soundfile if possible; falls back to torchaudio for formats soundfile can't read.
    Returns: (mono_wav, sr, channels, duration_s)
    """
    try:
        wav, sr = sf.read(str(path), always_2d=True)  # [T, C]
        wav = wav.astype(np.float32, copy=False)
        channels = int(wav.shape[1])
        duration_s = float(wav.shape[0]) / float(sr)
    except Exception:
        wav_t, sr = torchaudio.load(str(path))       # [C, T]
        wav_t = wav_t.to(torch.float32)
        channels = int(wav_t.shape[0])
        duration_s = float(wav_t.shape[1]) / float(sr)
        wav = wav_t.transpose(0, 1).cpu().numpy()    # [T, C]

    mono = wav.mean(axis=1) if wav.ndim == 2 and wav.shape[1] > 1 else wav.reshape(-1)
    return mono.astype(np.float32, copy=False), int(sr), channels, duration_s


def resample_to_16k(mono: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return mono
    x = torch.from_numpy(mono).to(torch.float32).unsqueeze(0)  # [1, T]
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
    with torch.no_grad():
        y = resampler(x).squeeze(0).cpu().numpy()
    return y.astype(np.float32, copy=False)


def process_file(file_path: Path, in_dir: Path, model, device: torch.device) -> FileResult:
    rel = str(file_path.relative_to(in_dir))
    try:
        mono, sr, ch, dur = load_audio(file_path)
        mono_16k = resample_to_16k(mono, sr)
        
        # Use the provided model
        model.eval()
        
        with torch.no_grad():
            x = torch.from_numpy(mono_16k).to(device).unsqueeze(0)
            mos = model.predict(data=x, sr=TARGET_SR, num_workers=0)
            if isinstance(mos, torch.Tensor):
                mos = float(mos.item())
            else:
                mos = float(np.asarray(mos).item())

        return FileResult(
            rel_path=rel,
            ok=True,
            mos=mos,
            mos_0_100=mos_to_0_100(mos),
            duration_s=dur,
            input_sr=sr,
            channels=ch,
            error=None,
        )
    except Exception as e:
        return FileResult(
            rel_path=rel,
            ok=False,
            mos=None,
            mos_0_100=None,
            duration_s=None,
            input_sr=None,
            channels=None,
            error=str(e),
        )


def compute_tables(df: pd.DataFrame) -> Dict[str, str]:
    ok_df = df[df["ok"] == True].copy()

    overall = {
        "files_total": len(df),
        "files_scored": int(ok_df.shape[0]),
        "files_failed": int((df["ok"] == False).sum()),
        "mos_mean": float(ok_df["mos"].mean()) if len(ok_df) else math.nan,
        "mos_median": float(ok_df["mos"].median()) if len(ok_df) else math.nan,
        "mos_std": float(ok_df["mos"].std(ddof=1)) if len(ok_df) > 1 else math.nan,
        "duration_hours": float(ok_df["duration_s"].sum() / 3600.0) if len(ok_df) else 0.0,
    }

    tables = {
        "overall": tabulate([[k, v] for k, v in overall.items()],
                            headers=["metric", "value"], tablefmt="github")
    }

    if len(ok_df):
        top = ok_df.sort_values("mos", ascending=False).head(10)[["rel_path", "mos", "mos_0_100", "duration_s"]]
        bottom = ok_df.sort_values("mos", ascending=True).head(10)[["rel_path", "mos", "mos_0_100", "duration_s"]]
        tables["top_10"] = tabulate(top, headers="keys", tablefmt="github", showindex=False)
        tables["bottom_10"] = tabulate(bottom, headers="keys", tablefmt="github", showindex=False)

    return tables


def write_html(df: pd.DataFrame, out_path: Path):
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>UTMOSv2 Folder Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    h1 {{ margin: 0 0 10px 0; }}
    .meta {{ color: #444; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f6f6f6; position: sticky; top: 0; }}
    tr:nth-child(even) {{ background: #fcfcfc; }}
  </style>
</head>
<body>
  <h1>UTMOSv2 Folder Report</h1>
  <div class="meta">Generated: {datetime.now().isoformat(timespec="seconds")}</div>
  {df.to_html(index=False, escape=True)}
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with audio files")
    ap.add_argument("--output_dir", required=True, help="Where to write results")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N files")
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_audio_files(in_dir, recursive=args.recursive)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    if not files:
        logger.warning("No audio files found in the input directory.")
        return

    # UTMOSv2 model (pretrained)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = utmosv2.create_model(pretrained=True).to(device)

    results: List[FileResult] = []
    # Temporarily disable multiprocessing due to Windows pickle issues
    for f in tqdm(files, desc="Scoring (UTMOSv2)"):
        results.append(process_file(f, in_dir, model, device))

    df = pd.DataFrame([asdict(r) for r in results])

    # Write outputs
    csv_path = out_dir / "results.csv"
    md_path = out_dir / "summary.md"
    html_path = out_dir / "report.html"

    df.to_csv(csv_path, index=False)

    tables = compute_tables(df)
    md = []
    md.append("# UTMOSv2 folder report\n\n")
    md.append(f"- Input: `{in_dir}`\n")
    md.append(f"- Recursive: `{args.recursive}`\n")
    md.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`\n\n")
    md.append("## Overall\n\n" + tables["overall"] + "\n\n")
    if "top_10" in tables:
        md.append("## Top 10 (highest MOS)\n\n" + tables["top_10"] + "\n\n")
        md.append("## Bottom 10 (lowest MOS)\n\n" + tables["bottom_10"] + "\n\n")

    fail_df = df[df["ok"] == False][["rel_path", "error"]]
    if len(fail_df):
        md.append("## Failures\n\n" + tabulate(fail_df, headers="keys", tablefmt="github", showindex=False) + "\n\n")

    md_path.write_text("".join(md), encoding="utf-8")
    write_html(df, html_path)

    logger.info("Done.")
    logger.info(f"Wrote:\n  {csv_path}\n  {md_path}\n  {html_path}")


if __name__ == "__main__":
    main()
