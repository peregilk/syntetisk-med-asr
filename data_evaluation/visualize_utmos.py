#!/usr/bin/env python3
"""Visualize UTMOS results (reads results.csv and creates plots + HTML preview)."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def make_plots(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Histogram of MOS (1-5)
    plt.figure(figsize=(8, 4))
    sns.histplot(df["mos"].dropna(), bins=20, kde=True)
    plt.title("UTMOS MOS distribution")
    plt.xlabel("MOS (1-5)")
    plt.tight_layout()
    hist_path = out_dir / "mos_hist.png"
    plt.savefig(hist_path)
    plt.close()

    # Boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df["mos"].dropna())
    plt.title("MOS boxplot")
    plt.xlabel("MOS (1-5)")
    plt.tight_layout()
    box_path = out_dir / "mos_box.png"
    plt.savefig(box_path)
    plt.close()

    # MOS vs duration
    if "duration_s" in df.columns:
        plt.figure(figsize=(8, 4))
        sns.scatterplot(x=df["duration_s"], y=df["mos"])
        plt.xlabel("Duration (s)")
        plt.ylabel("MOS")
        plt.title("MOS vs Duration")
        plt.tight_layout()
        scatter_path = out_dir / "mos_vs_duration.png"
        plt.savefig(scatter_path)
        plt.close()
    else:
        scatter_path = None

    # Top / bottom N bar charts (use mos_0_100 if available otherwise mos)
    score_col = "mos_0_100" if "mos_0_100" in df.columns else "mos"

    top_n = df.sort_values(by=score_col, ascending=False).head(10)
    bottom_n = df.sort_values(by=score_col, ascending=True).head(10)

    plt.figure(figsize=(10, 4))
    sns.barplot(x=score_col, y="rel_path", data=top_n)
    plt.title("Top 10 files")
    plt.tight_layout()
    top_path = out_dir / "top_10.png"
    plt.savefig(top_path)
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.barplot(x=score_col, y="rel_path", data=bottom_n)
    plt.title("Bottom 10 files")
    plt.tight_layout()
    bottom_path = out_dir / "bottom_10.png"
    plt.savefig(bottom_path)
    plt.close()

    return {
        "hist": hist_path,
        "box": box_path,
        "scatter": scatter_path,
        "top": top_path,
        "bottom": bottom_path,
    }


def make_html(image_paths: dict, out_path: Path, df: pd.DataFrame):
    lines = [
        "<html>",
        "<head><meta charset='utf-8'><title>UTMOS visualization</title></head>",
        "<body>",
        "<h1>UTMOS Results Visualization</h1>",
    ]

    # Basic stats
    stats = df["mos"].describe().to_frame().to_html()
    lines.append("<h2>Basic MOS stats</h2>")
    lines.append(stats)

    # Images
    for label, p in image_paths.items():
        if p is None:
            continue
        lines.append(f"<h3>{label}</h3>")
        lines.append(f"<img src='{p.name}' style='max-width:100%;height:auto'/>")

    # Top 10 table
    if not df.empty:
        top10 = df.sort_values(by=("mos_0_100" if "mos_0_100" in df.columns else "mos"), ascending=False).head(10)
        lines.append("<h2>Top 10</h2>")
        lines.append(top10[["rel_path", "mos"]].to_html(index=False))

    lines.append("</body></html>")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data_evaluation/utmos_output/results.csv", help="Path to results.csv")
    ap.add_argument("--out_dir", default="data_evaluation/utmos_output", help="Where to write images and HTML")
    args = ap.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.out_dir)

    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")

    df = pd.read_csv(results_path)

    images = make_plots(df, out_dir)
    html_path = out_dir / "visualization.html"
    make_html(images, html_path, df)

    print("Wrote:")
    for v in images.values():
        if v:
            print(" ", v)
    print(" ", html_path)


if __name__ == "__main__":
    main()
